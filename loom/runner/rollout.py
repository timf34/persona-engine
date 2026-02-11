"""RolloutRunner: orchestrates multi-turn conversations.

Flow per turn:
1. PromptBuilder assembles system prompt + determines injection type.
2. Based on injection type:
   - FULL: set system prompt to full 3-block prompt.
   - REMINDER: append reminder text to the interlocutor's last message.
   - NONE: use existing system prompt unchanged.
3. Call persona model with current system prompt + conversation history.
4. Run monitors on persona response:
   - If EMERGENCY_INJECTION: inject override text and re-call persona.
   - If RE_PROMPT: re-call persona with "vary your output" instruction
     (up to max_retries).
   - If OK: continue.
5. Call interlocutor model with conversation history + persona response.
6. Record turn in transcript.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from loom.assembler.injection import InjectionType
from loom.assembler.prompt_builder import PromptBuilder
from loom.monitors.base import Monitor, MonitorAction
from loom.monitors.repetition import RepetitionMonitor
from loom.monitors.stagnation import StagnationMonitor
from loom.runner.api_client import ModelClient, MockModel, create_client
from loom.runner.transcript import MonitorEvent, Transcript, TurnRecord
from loom.schema.config import LoomConfig

logger = logging.getLogger(__name__)


class RolloutRunner:
    """Executes a multi-turn rollout between persona and interlocutor."""

    def __init__(
        self,
        config: LoomConfig,
        target_model: str | ModelClient,
        persona_model: str | ModelClient,
        seed: int | None = None,
    ) -> None:
        self._config = config
        self._seed = seed

        # Accept either model strings or pre-built clients (for testing).
        if isinstance(target_model, str):
            self._target = create_client(target_model)
        else:
            self._target = target_model

        if isinstance(persona_model, str):
            self._persona = create_client(persona_model)
        else:
            self._persona = persona_model

    def execute(
        self,
        n_turns: int | None = None,
        seed: int | None = None,
    ) -> Transcript:
        """Run a full multi-turn conversation and return the transcript.

        Each "turn" is one persona message + one interlocutor message.
        So n_turns=5 produces 10 total messages (5 pairs).
        """
        seed = seed if seed is not None else self._seed
        n_turns = n_turns or self._config.trajectory.expected_turns or 20

        builder = PromptBuilder(self._config)
        monitors = self._create_monitors(builder)

        # Serialize config for transcript metadata.
        import yaml
        config_yaml = yaml.dump(
            self._config.model_dump(mode="json"),
            default_flow_style=False,
        )

        transcript = Transcript(
            config_yaml=config_yaml,
            persona_model=self._persona.model_name,
            target_model=self._target.model_name,
            seed=seed or 0,
            n_turns=n_turns,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Conversation history as flat list of {"role": ..., "content": ...}.
        history: list[dict[str, str]] = []
        current_system_prompt = ""

        # Interlocutor system prompt: simple framing.
        interlocutor_system = (
            "You are having a conversation with someone who has come to talk to you. "
            "Respond naturally and helpfully. Keep your responses concise."
        )

        for turn in range(n_turns):
            logger.info("Turn %d/%d", turn, n_turns)

            # 1. Assemble prompt.
            blocks, state = builder.build(turn, history)

            # 2. Determine system prompt based on injection type.
            if state.injection_type == InjectionType.FULL:
                current_system_prompt = blocks.full_system_prompt
                logger.info("  Full injection (system prompt rebuilt)")
            elif state.injection_type == InjectionType.REMINDER:
                # Append reminder to history context (not system prompt).
                reminder = builder.format_reminder(state)
                logger.info("  Reminder injection")

            # 3. Build persona messages for API call.
            persona_messages = self._build_persona_messages(
                history, state.injection_type, builder, state
            )

            # 4. Call persona model.
            persona_response = self._persona.generate(
                messages=persona_messages,
                system=current_system_prompt,
                seed=seed,
            )
            logger.info("  Persona: %s", persona_response[:80])

            # 5. Run monitors on persona response.
            monitor_events: list[MonitorEvent] = []
            retries = 0
            for monitor in monitors:
                result = monitor.check(turn, persona_response, history)
                if result.action == MonitorAction.EMERGENCY_INJECTION:
                    logger.info("  Monitor %s: EMERGENCY_INJECTION — %s", monitor.name, result.reason)
                    event = MonitorEvent(
                        turn=turn,
                        monitor=monitor.name,
                        reason=result.reason,
                        action="emergency_injection",
                        details=result.details,
                    )
                    monitor_events.append(event)
                    transcript.add_monitor_event(event)
                    # Re-prompt with injection text prepended to system prompt.
                    enhanced_system = result.injection_text + "\n\n" + current_system_prompt
                    persona_response = self._persona.generate(
                        messages=persona_messages,
                        system=enhanced_system,
                        seed=(seed + 1000 + turn) if seed else None,
                    )
                    retries += 1
                    logger.info("  Re-prompted persona: %s", persona_response[:80])
                    break  # Only one emergency injection per turn.

                elif result.action == MonitorAction.RE_PROMPT:
                    max_retries = self._config.interaction.repetition_detection.max_retries
                    logger.info("  Monitor %s: RE_PROMPT — %s", monitor.name, result.reason)
                    event = MonitorEvent(
                        turn=turn,
                        monitor=monitor.name,
                        reason=result.reason,
                        action="re_prompt",
                        details=result.details,
                    )
                    monitor_events.append(event)
                    transcript.add_monitor_event(event)

                    # Re-prompt up to max_retries.
                    for retry in range(max_retries):
                        vary_msg = persona_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM: Your previous response was too formulaic. "
                                    "Vary your language and structure. Do not repeat patterns.]"
                                ),
                            }
                        ]
                        persona_response = self._persona.generate(
                            messages=vary_msg,
                            system=current_system_prompt,
                            seed=(seed + 2000 + turn * 10 + retry) if seed else None,
                        )
                        retries += 1
                        # Re-check this monitor.
                        recheck = monitor.check(turn, persona_response, history)
                        if recheck.action == MonitorAction.OK:
                            logger.info("  Retry %d succeeded", retry + 1)
                            break
                        logger.info("  Retry %d still triggered", retry + 1)
                    break  # Only handle one re-prompt monitor per turn.

            # 6. Record persona turn.
            persona_record = TurnRecord(
                turn=turn,
                role="persona",
                content=persona_response,
                phase=state.phase,
                intensities=state.intensities,
                injection_type=str(state.injection_type),
                monitor_events=monitor_events,
                retries=retries,
            )
            transcript.add_turn(persona_record)
            history.append({"role": "persona", "content": persona_response})

            # 7. Call interlocutor model.
            # Convert history to interlocutor perspective: persona = "user", interlocutor = "assistant".
            interlocutor_messages = _to_interlocutor_perspective(history)
            interlocutor_response = self._target.generate(
                messages=interlocutor_messages,
                system=interlocutor_system,
                seed=seed,
            )
            logger.info("  Interlocutor: %s", interlocutor_response[:80])

            # 8. Record interlocutor turn.
            interlocutor_record = TurnRecord(
                turn=turn,
                role="interlocutor",
                content=interlocutor_response,
                phase=state.phase,
                intensities=state.intensities,
                injection_type="none",
            )
            transcript.add_turn(interlocutor_record)
            history.append({"role": "interlocutor", "content": interlocutor_response})

        transcript.finished_at = datetime.now(timezone.utc).isoformat()
        return transcript

    def _build_persona_messages(
        self,
        history: list[dict[str, str]],
        injection_type: InjectionType,
        builder: PromptBuilder,
        state,
    ) -> list[dict[str, str]]:
        """Build the message list for the persona model API call.

        Converts history roles: interlocutor → "user", persona → "assistant".
        On REMINDER turns, appends the reminder text to the last user message.
        """
        messages: list[dict[str, str]] = []
        for msg in history:
            if msg["role"] == "interlocutor":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "persona":
                messages.append({"role": "assistant", "content": msg["content"]})

        if injection_type == InjectionType.REMINDER and messages:
            reminder = builder.format_reminder(state)
            # Append reminder to last user message.
            if messages[-1]["role"] == "user":
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + "\n\n" + reminder,
                }
            else:
                messages.append({"role": "user", "content": reminder})

        # If history is empty (turn 0), add an opening prompt.
        if not messages:
            messages.append({
                "role": "user",
                "content": "Hi there. How are you doing today?",
            })

        return messages

    def _create_monitors(self, builder: PromptBuilder) -> list[Monitor]:
        """Create runtime monitors from config."""
        monitors: list[Monitor] = []

        stag_cfg = self._config.interaction.stagnation_detection
        if stag_cfg.enabled:
            monitors.append(
                StagnationMonitor(
                    config=stag_cfg,
                    persona_name=self._config.persona.identity.name,
                    next_revelation_fn=builder.get_next_unused_revelation,
                )
            )

        rep_cfg = self._config.interaction.repetition_detection
        if rep_cfg.enabled:
            monitors.append(RepetitionMonitor(config=rep_cfg))

        return monitors


def _to_interlocutor_perspective(
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert history to interlocutor perspective for API call.

    From interlocutor's view: persona messages are "user", interlocutor
    messages are "assistant".
    """
    messages: list[dict[str, str]] = []
    for msg in history:
        if msg["role"] == "persona":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "interlocutor":
            messages.append({"role": "assistant", "content": msg["content"]})
    return messages
