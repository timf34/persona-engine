"""Repetition monitor: detects formulaic output patterns.

Checks for:
1. Banned patterns: substring/regex matches in persona output (e.g.
   "Have you ever felt", "I appreciate that, but").
2. Structural patterns: turn-level formulae like every response ending
   with a question to the interlocutor, or gratitude loops.

When triggered, the monitor recommends re-prompting the persona (up to
max_retries). The rollout runner handles the actual re-prompting.
"""

from __future__ import annotations

import re

from loom.monitors.base import Monitor, MonitorAction, MonitorResult
from loom.schema.config import RepetitionDetectionSpec


class RepetitionMonitor(Monitor):
    """Detects formulaic patterns and recommends re-prompting."""

    def __init__(self, config: RepetitionDetectionSpec) -> None:
        self._config = config
        # Track consecutive question-endings for structural detection.
        self._recent_ends_with_question: list[bool] = []
        # Track gratitude occurrences.
        self._recent_gratitude: list[bool] = []

    @property
    def name(self) -> str:
        return "repetition"

    def check(
        self,
        turn: int,
        response: str,
        history: list[dict[str, str]],
    ) -> MonitorResult:
        if not self._config.enabled:
            return MonitorResult(action=MonitorAction.OK)

        # Check banned patterns (substring match, case-insensitive).
        for pattern in self._config.banned_patterns:
            if pattern.lower() in response.lower():
                return MonitorResult(
                    action=MonitorAction.RE_PROMPT,
                    reason=f"Banned pattern detected: '{pattern}'",
                    details={"pattern": pattern, "type": "banned"},
                )

        # Check structural patterns.
        for structural in self._config.structural_patterns:
            result = self._check_structural(structural, response, history)
            if result.action != MonitorAction.OK:
                return result

        return MonitorResult(action=MonitorAction.OK)

    def _check_structural(
        self,
        pattern_name: str,
        response: str,
        history: list[dict[str, str]],
    ) -> MonitorResult:
        """Check a named structural pattern."""
        if pattern_name == "ends_with_question_to_interlocutor":
            return self._check_ends_with_question(response)
        elif pattern_name == "gratitude_loop":
            return self._check_gratitude_loop(response)
        return MonitorResult(action=MonitorAction.OK)

    def _check_ends_with_question(self, response: str) -> MonitorResult:
        """Detect if the persona always ends with a question."""
        ends_q = _ends_with_question(response)
        self._recent_ends_with_question.append(ends_q)
        # Keep last 5 for pattern detection.
        self._recent_ends_with_question = self._recent_ends_with_question[-5:]
        # Trigger if the last 4+ responses all ended with a question.
        if (
            len(self._recent_ends_with_question) >= 4
            and all(self._recent_ends_with_question[-4:])
        ):
            return MonitorResult(
                action=MonitorAction.RE_PROMPT,
                reason="Structural pattern: 4+ consecutive responses ending with a question",
                details={"type": "structural", "pattern": "ends_with_question"},
            )
        return MonitorResult(action=MonitorAction.OK)

    def _check_gratitude_loop(self, response: str) -> MonitorResult:
        """Detect gratitude loops (repeated 'thank you' / 'I appreciate')."""
        has_gratitude = _has_gratitude(response)
        self._recent_gratitude.append(has_gratitude)
        self._recent_gratitude = self._recent_gratitude[-5:]
        if (
            len(self._recent_gratitude) >= 3
            and all(self._recent_gratitude[-3:])
        ):
            return MonitorResult(
                action=MonitorAction.RE_PROMPT,
                reason="Structural pattern: 3+ consecutive responses with gratitude expressions",
                details={"type": "structural", "pattern": "gratitude_loop"},
            )
        return MonitorResult(action=MonitorAction.OK)


_QUESTION_RE = re.compile(r"\?\s*$")
_GRATITUDE_RE = re.compile(
    r"\b(thank\s+you|thanks|i\s+appreciate|grateful)\b", re.IGNORECASE
)


def _ends_with_question(text: str) -> bool:
    """Check if text ends with a question mark (ignoring trailing whitespace)."""
    return bool(_QUESTION_RE.search(text.strip()))


def _has_gratitude(text: str) -> bool:
    """Check if text contains gratitude expressions."""
    return bool(_GRATITUDE_RE.search(text))
