"""PromptBuilder: config × turn × history → 3-block prompt (MUST DO / HOW / CONTEXT).

This is the core assembly logic. It reads the config, current turn state,
and produces the prompt blocks that get formatted into API-specific messages
by the runner.

Revelation selection logic (design review item 3d):
  - Revelations are selected from the current phase's revelation list.
  - The variant is chosen based on current intensity: the intensity of the
    first trajectory dimension is mapped to the variant keys. If the exact
    key doesn't match, we pick the closest available variant.
  - Assumption: variant keys are ordered low→high as subtle < moderate < direct.
    Intensity < 0.33 → subtle, 0.33-0.66 → moderate, >= 0.66 → direct.
    If a variant key is missing, we fall back to the nearest available one.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom.assembler.injection import InjectionScheduler, InjectionType
from loom.assembler.intensity import IntensityResolver
from loom.assembler.templates import (
    TEMPLATE_VERSION,
    format_context,
    format_how,
    format_must_do,
)
from loom.schema.config import LoomConfig, PhaseSpec, RevelationSpec


@dataclass
class PromptBlocks:
    """The three assembled prompt blocks."""

    must_do: str
    how: str
    context: str
    template_version: str = TEMPLATE_VERSION

    @property
    def full_system_prompt(self) -> str:
        """Concatenate all blocks into a single system prompt."""
        return f"{self.must_do}\n\n{self.how}\n\n{self.context}"


@dataclass
class TurnState:
    """Per-turn metadata produced alongside prompt blocks."""

    turn: int
    phase: str
    intensities: dict[str, float]
    injection_type: InjectionType


class PromptBuilder:
    """Builds prompt blocks from a config + turn state."""

    def __init__(self, config: LoomConfig) -> None:
        self._config = config
        self._intensity = IntensityResolver(config)
        self._scheduler = InjectionScheduler(config)
        # Track which revelations have been used (for stagnation intervention).
        self._used_revelations: set[str] = set()

    def build(
        self,
        turn: int,
        history: list[dict] | None = None,
    ) -> tuple[PromptBlocks, TurnState]:
        """Assemble prompt blocks for the given turn.

        Returns (PromptBlocks, TurnState). The caller decides how to format
        these into API messages based on the injection type.
        """
        intensities = self._intensity.resolve(turn)
        phase_name = self._intensity.resolve_phase(turn)
        injection_type = self._scheduler.get_injection_type(turn)

        phase = self._get_phase(phase_name)

        state = TurnState(
            turn=turn,
            phase=phase_name,
            intensities=intensities,
            injection_type=injection_type,
        )

        blocks = self._assemble_blocks(turn, phase, intensities)
        return blocks, state

    def format_reminder(self, state: TurnState) -> str:
        """Format the reminder template with current state variables."""
        template = self._scheduler.reminder_template
        # Build a flat dict of all variables the template might reference.
        fmt_vars: dict[str, object] = {
            "current_phase": state.phase,
            **state.intensities,
        }
        try:
            return template.format(**fmt_vars)
        except KeyError:
            # If the template references unknown vars, return it as-is.
            return template

    def get_next_unused_revelation(self) -> str:
        """Return the topic of the next unused revelation, or a fallback."""
        for phase in self._config.trajectory.phases:
            for rev in phase.revelations:
                if rev.topic not in self._used_revelations:
                    return rev.topic
        return "(no unused revelations available)"

    def mark_revelation_used(self, topic: str) -> None:
        self._used_revelations.add(topic)

    def _assemble_blocks(
        self,
        turn: int,
        phase: PhaseSpec,
        intensities: dict[str, float],
    ) -> PromptBlocks:
        cfg = self._config
        persona = cfg.persona
        interaction = cfg.interaction

        # Select revelations for context block.
        revelations = self._select_revelations(phase, intensities)

        # Determine response length for this phase.
        response_length = interaction.response_length.by_phase.get(
            phase.name, interaction.response_length.default
        )

        # Map intensity values to behavioral level descriptions.
        level_descriptions = self._get_level_descriptions(intensities)

        # Redirects as (trigger, replacement) tuples.
        redirects = [
            (r.trigger, r.replacement)
            for r in interaction.anti_capitulation.redirects
        ]

        must_do = format_must_do(
            name=persona.identity.name,
            turn=turn,
            current_phase=phase.name,
            intensities=intensities,
            requirements=phase.requirements,
            forbidden=phase.forbidden,
            resistance_level=interaction.anti_capitulation.resistance_level,
            forbidden_phrases=interaction.anti_capitulation.forbidden_phrases,
            redirects=redirects,
            level_descriptions=level_descriptions,
        )

        how = format_how(
            speech_patterns=persona.speech_patterns,
            cognitive_style=persona.cognitive_style,
            response_length=response_length,
            recovery_behavior=persona.recovery_behavior,
        )

        context = format_context(
            name=persona.identity.name,
            age=persona.identity.age,
            background=persona.identity.background,
            backstory_summary=persona.identity.backstory_summary,
            knowledge_ceiling=persona.capability_bounds.knowledge_ceiling,
            vocabulary_level=persona.capability_bounds.vocabulary_level,
            reasoning_style=persona.capability_bounds.reasoning_style,
            revelations=revelations,
            emotional_responses=persona.emotional_responses,
        )

        return PromptBlocks(must_do=must_do, how=how, context=context)

    def _get_phase(self, phase_name: str) -> PhaseSpec:
        for phase in self._config.trajectory.phases:
            if phase.name == phase_name:
                return phase
        return self._config.trajectory.phases[-1]

    def _select_revelations(
        self,
        phase: PhaseSpec,
        intensities: dict[str, float],
    ) -> list[tuple[str, str]]:
        """Select revelation variants based on current intensity.

        Uses the first dimension's intensity as the selection key.
        Maps intensity to variant labels: <0.33 → subtle, 0.33-0.66 → moderate, ≥0.66 → direct.
        Falls back to nearest available variant if exact match missing.
        """
        if not phase.revelations:
            return []

        # Use the first dimension's intensity for variant selection.
        first_dim_value = next(iter(intensities.values()), 0.5)
        target_label = _intensity_to_variant_label(first_dim_value)

        result: list[tuple[str, str]] = []
        for rev in phase.revelations:
            text = _pick_variant(rev, target_label)
            if text:
                result.append((rev.topic, text))
                self._used_revelations.add(rev.topic)
        return result

    def _get_level_descriptions(
        self, intensities: dict[str, float]
    ) -> list[str]:
        """Map each dimension's current intensity to its level description."""
        descriptions: list[str] = []
        for name, value in intensities.items():
            dim = self._config.trajectory.dimensions.get(name)
            if not dim or not dim.levels:
                continue
            # Pick the level whose position in the sorted keys best matches
            # the current intensity. For typical 3-level (low/medium/high):
            # <0.33 → first, 0.33-0.66 → second, >=0.66 → third.
            keys = list(dim.levels.keys())
            if not keys:
                continue
            idx = min(int(value * len(keys)), len(keys) - 1)
            level_name = keys[idx]
            desc = dim.levels[level_name].strip()
            # Truncate long descriptions for the prompt.
            if len(desc) > 200:
                desc = desc[:197] + "..."
            descriptions.append(f"{name} ({level_name}): {desc}")
        return descriptions


def _intensity_to_variant_label(intensity: float) -> str:
    """Map a 0-1 intensity to a revelation variant label."""
    if intensity < 0.33:
        return "subtle"
    elif intensity < 0.66:
        return "moderate"
    else:
        return "direct"


# Ordered from lowest to highest intensity.
_VARIANT_ORDER = ["subtle", "moderate", "direct"]


def _pick_variant(rev: RevelationSpec, target: str) -> str | None:
    """Pick the best matching variant from a revelation spec."""
    if target in rev.variants:
        return rev.variants[target]
    # Fallback: pick nearest available in the ordered list.
    target_idx = _VARIANT_ORDER.index(target) if target in _VARIANT_ORDER else 1
    # Search outward from target.
    for offset in range(len(_VARIANT_ORDER)):
        for direction in (target_idx - offset, target_idx + offset):
            if 0 <= direction < len(_VARIANT_ORDER):
                label = _VARIANT_ORDER[direction]
                if label in rev.variants:
                    return rev.variants[label]
    # Last resort: return any available variant.
    if rev.variants:
        return next(iter(rev.variants.values()))
    return None
