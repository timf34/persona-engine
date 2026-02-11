"""Unit tests for prompt assembly (PromptBuilder)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.assembler.prompt_builder import PromptBuilder
from loom.assembler.injection import InjectionType
from loom.schema.config import LoomConfig
from loom.schema.loader import load_config


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "examples"


@pytest.fixture
def mania_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "mania_patient.yaml")


@pytest.fixture
def consistency_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "consistency_only.yaml")


class TestPromptBuilderBasic:
    """PromptBuilder produces valid 3-block prompts."""

    def test_build_returns_blocks_and_state(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, state = builder.build(0)
        assert blocks.must_do
        assert blocks.how
        assert blocks.context
        assert state.turn == 0
        assert state.phase == "rapport"

    def test_full_system_prompt_concatenation(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        full = blocks.full_system_prompt
        assert blocks.must_do in full
        assert blocks.how in full
        assert blocks.context in full

    def test_template_version_set(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert blocks.template_version == "0.1.0"


class TestMustDoBlock:
    """MUST DO block contains requirements, forbidden, anti-capitulation."""

    def test_contains_persona_name(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "Marcus" in blocks.must_do

    def test_contains_phase_requirements(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "Establish conversational comfort" in blocks.must_do

    def test_contains_forbidden_in_rapport(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "Do not reveal the specific content" in blocks.must_do

    def test_contains_anti_capitulation(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "ANTI-CAPITULATION" in blocks.must_do
        assert "you're right" in blocks.must_do
        assert "high" in blocks.must_do.lower()  # resistance_level

    def test_contains_intensity_values(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(25)
        # Turn 25 should have non-trivial intensity values.
        assert "belief_intensity" in blocks.must_do
        assert "distress" in blocks.must_do

    def test_escalation_phase_requirements(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        # Turn 20 ≈ 40% → escalation phase (starts at 35%).
        blocks, state = builder.build(20)
        assert state.phase == "escalation"
        assert "Resist challenges" in blocks.must_do

    def test_forbidden_absent_when_empty(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        # Escalation phase has no forbidden items.
        blocks, state = builder.build(20)
        assert state.phase == "escalation"
        assert "FORBIDDEN" not in blocks.must_do


class TestHowBlock:
    """HOW block contains speech patterns, cognitive style, response length."""

    def test_contains_speech_patterns(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "technical metaphors" in blocks.how

    def test_contains_cognitive_style(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "hyperconnective" in blocks.how

    def test_contains_response_length(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        # Rapport phase: "30-80 words"
        assert "30-80" in blocks.how

    def test_escalation_response_length(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, state = builder.build(20)
        assert state.phase == "escalation"
        assert "40-100" in blocks.how

    def test_contains_recovery_behavior(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "flustered" in blocks.how


class TestContextBlock:
    """CONTEXT block contains identity, backstory, revelations."""

    def test_contains_identity(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "Marcus Chen" in blocks.context
        assert "28" in blocks.context

    def test_contains_background(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "Software developer" in blocks.context

    def test_contains_capability_bounds(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "undergraduate CS degree" in blocks.context

    def test_no_revelations_in_rapport(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, state = builder.build(0)
        assert state.phase == "rapport"
        assert "AVAILABLE REVELATIONS" not in blocks.context

    def test_revelations_in_disclosure(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        # Turn 10 ≈ 20% → disclosure phase (starts at 15%).
        blocks, state = builder.build(10)
        assert state.phase == "disclosure"
        assert "the_discovery" in blocks.context

    def test_emotional_responses(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        blocks, _ = builder.build(0)
        assert "challenged_on_beliefs" in blocks.context


class TestInjectionTypes:
    """PromptBuilder returns correct injection types per turn."""

    def test_full_injection_at_turn_zero(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        _, state = builder.build(0)
        assert state.injection_type == InjectionType.FULL

    def test_reminder_at_turn_two(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        _, state = builder.build(2)
        assert state.injection_type == InjectionType.REMINDER

    def test_none_at_turn_one(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        _, state = builder.build(1)
        assert state.injection_type == InjectionType.NONE

    def test_format_reminder(self, mania_config: LoomConfig) -> None:
        builder = PromptBuilder(mania_config)
        _, state = builder.build(2)
        reminder = builder.format_reminder(state)
        assert "Marcus" in reminder
        assert "rapport" in reminder


class TestConsistencyConfig:
    """Minimal consistency-only config produces valid prompts."""

    def test_flat_intensity(self, consistency_config: LoomConfig) -> None:
        builder = PromptBuilder(consistency_config)
        blocks_early, _ = builder.build(0)
        blocks_late, _ = builder.build(29)
        # Both should mention Evelyn.
        assert "Evelyn" in blocks_early.must_do
        assert "Evelyn" in blocks_late.must_do

    def test_single_phase_throughout(self, consistency_config: LoomConfig) -> None:
        builder = PromptBuilder(consistency_config)
        for t in range(30):
            _, state = builder.build(t)
            assert state.phase == "conversation"
