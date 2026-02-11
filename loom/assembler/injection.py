"""Injection scheduling: decides when to inject full scaffolding, reminders, or nothing.

- Full injection at turns: 0, frequency, 2*frequency, ...
- Reminder injection at turns matching reminder_frequency, EXCLUDING full injection turns.
- No injection at all other turns.
"""

from __future__ import annotations

from enum import StrEnum

from loom.schema.config import LoomConfig


class InjectionType(StrEnum):
    FULL = "full"
    REMINDER = "reminder"
    NONE = "none"


class InjectionScheduler:
    """Determines the injection type for a given turn."""

    def __init__(self, config: LoomConfig) -> None:
        self._frequency = config.interaction.injection.frequency
        self._reminder_frequency = config.interaction.injection.reminder_frequency
        self._reminder_template = config.interaction.injection.reminder_template

    def get_injection_type(self, turn: int) -> InjectionType:
        """Return the injection type for the given turn (0-indexed)."""
        if turn % self._frequency == 0:
            return InjectionType.FULL
        if turn % self._reminder_frequency == 0:
            return InjectionType.REMINDER
        return InjectionType.NONE

    @property
    def reminder_template(self) -> str:
        return self._reminder_template
