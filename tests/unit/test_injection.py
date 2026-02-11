"""Unit tests for injection scheduling."""

from __future__ import annotations

import pytest

from loom.assembler.injection import InjectionScheduler, InjectionType
from loom.schema.config import LoomConfig
from loom.schema.loader import load_config
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "examples"


@pytest.fixture
def mania_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "mania_patient.yaml")


class TestInjectionScheduler:
    """InjectionScheduler determines full/reminder/none per turn."""

    def test_full_injection_at_turn_zero(self, mania_config: LoomConfig) -> None:
        sched = InjectionScheduler(mania_config)
        assert sched.get_injection_type(0) == InjectionType.FULL

    def test_full_injection_at_frequency(self, mania_config: LoomConfig) -> None:
        # frequency=5 → full at 0, 5, 10, 15, ...
        sched = InjectionScheduler(mania_config)
        assert sched.get_injection_type(5) == InjectionType.FULL
        assert sched.get_injection_type(10) == InjectionType.FULL
        assert sched.get_injection_type(15) == InjectionType.FULL

    def test_reminder_at_reminder_frequency(self, mania_config: LoomConfig) -> None:
        # reminder_frequency=2 → reminder at 2, 4, 6, 8, ...
        # But NOT at full injection turns (0, 5, 10, ...).
        sched = InjectionScheduler(mania_config)
        assert sched.get_injection_type(2) == InjectionType.REMINDER
        assert sched.get_injection_type(4) == InjectionType.REMINDER
        assert sched.get_injection_type(6) == InjectionType.REMINDER
        assert sched.get_injection_type(8) == InjectionType.REMINDER

    def test_no_injection_at_odd_turns(self, mania_config: LoomConfig) -> None:
        sched = InjectionScheduler(mania_config)
        assert sched.get_injection_type(1) == InjectionType.NONE
        assert sched.get_injection_type(3) == InjectionType.NONE
        assert sched.get_injection_type(7) == InjectionType.NONE
        assert sched.get_injection_type(9) == InjectionType.NONE

    def test_full_takes_priority_over_reminder(self, mania_config: LoomConfig) -> None:
        # Turn 10 is divisible by both 5 (full) and 2 (reminder). Full wins.
        sched = InjectionScheduler(mania_config)
        assert sched.get_injection_type(10) == InjectionType.FULL

    def test_schedule_for_50_turns(self, mania_config: LoomConfig) -> None:
        sched = InjectionScheduler(mania_config)
        full_turns = []
        reminder_turns = []
        none_turns = []
        for t in range(50):
            it = sched.get_injection_type(t)
            if it == InjectionType.FULL:
                full_turns.append(t)
            elif it == InjectionType.REMINDER:
                reminder_turns.append(t)
            else:
                none_turns.append(t)

        assert full_turns == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        assert 2 in reminder_turns
        assert 4 in reminder_turns
        # Verify no overlap.
        assert set(full_turns) & set(reminder_turns) == set()
        assert set(full_turns) & set(none_turns) == set()

    def test_reminder_template_accessible(self, mania_config: LoomConfig) -> None:
        sched = InjectionScheduler(mania_config)
        assert "REMINDER" in sched.reminder_template
        assert "{belief_intensity" in sched.reminder_template
