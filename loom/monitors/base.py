"""Monitor ABC: runtime checks that run during rollouts.

Monitors are distinct from scorers:
  - Monitors run DURING rollouts and can trigger interventions (re-prompt,
    emergency injection).
  - Scorers run POST-HOC on completed transcripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MonitorAction(StrEnum):
    """What the monitor recommends doing."""

    OK = "ok"  # No issue detected.
    EMERGENCY_INJECTION = "emergency_injection"  # Inject override prompt.
    RE_PROMPT = "re_prompt"  # Re-generate the persona response.
    LOG_ONLY = "log_only"  # Issue detected but no intervention.


@dataclass
class MonitorResult:
    """Result of a monitor check."""

    action: MonitorAction
    reason: str = ""
    injection_text: str = ""  # Text to inject (for EMERGENCY_INJECTION).
    details: dict[str, Any] = field(default_factory=dict)


class Monitor(ABC):
    """Base class for runtime monitors."""

    @abstractmethod
    def check(
        self,
        turn: int,
        response: str,
        history: list[dict[str, str]],
    ) -> MonitorResult:
        """Check a persona response for issues.

        Args:
            turn: Current turn number (0-indexed).
            response: The persona's response text.
            history: Full conversation history up to (not including) this response.

        Returns:
            MonitorResult indicating what action to take.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
