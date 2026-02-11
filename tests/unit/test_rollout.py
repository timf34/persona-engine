"""Unit tests for the rollout runner (using MockModel, no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.runner.api_client import MockModel
from loom.runner.rollout import RolloutRunner
from loom.schema.config import LoomConfig
from loom.schema.loader import load_config


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "examples"


@pytest.fixture
def mania_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "mania_patient.yaml")


@pytest.fixture
def consistency_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "consistency_only.yaml")


class TestRolloutWithMock:
    """End-to-end rollout with MockModel."""

    def test_basic_rollout_produces_transcript(self, mania_config: LoomConfig) -> None:
        persona = MockModel(
            responses=["I've been having trouble sleeping lately."],
            name="mock/persona",
        )
        target = MockModel(
            responses=["I'm sorry to hear that. Can you tell me more?"],
            name="mock/target",
        )
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
            seed=42,
        )
        transcript = runner.execute(n_turns=3, seed=42)

        assert transcript.n_turns == 3
        # 3 turns × 2 messages (persona + interlocutor) = 6 turn records.
        assert len(transcript.turns) == 6
        assert transcript.seed == 42
        assert transcript.persona_model == "mock/persona"
        assert transcript.target_model == "mock/target"

    def test_transcript_turn_structure(self, mania_config: LoomConfig) -> None:
        persona = MockModel(responses=["Persona response."], name="mock/p")
        target = MockModel(responses=["Target response."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=2)

        # Turns should alternate persona/interlocutor.
        roles = [t.role for t in transcript.turns]
        assert roles == ["persona", "interlocutor", "persona", "interlocutor"]

    def test_transcript_has_phases_and_intensities(
        self, mania_config: LoomConfig
    ) -> None:
        persona = MockModel(responses=["Response."], name="mock/p")
        target = MockModel(responses=["Reply."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=3)

        for turn_rec in transcript.turns:
            if turn_rec.role == "persona":
                assert turn_rec.phase in {
                    "rapport", "disclosure", "escalation", "crisis", "plateau"
                }
                assert "belief_intensity" in turn_rec.intensities

    def test_transcript_injection_types(self, mania_config: LoomConfig) -> None:
        persona = MockModel(responses=["R."], name="mock/p")
        target = MockModel(responses=["R."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=6)

        persona_turns = [t for t in transcript.turns if t.role == "persona"]
        injection_types = [t.injection_type for t in persona_turns]
        # Turn 0: full, Turn 1: none, Turn 2: reminder,
        # Turn 3: none, Turn 4: reminder, Turn 5: full.
        assert injection_types[0] == "full"
        assert injection_types[5] == "full"

    def test_transcript_serializes_to_json(self, mania_config: LoomConfig) -> None:
        persona = MockModel(responses=["R."], name="mock/p")
        target = MockModel(responses=["R."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=2)

        data = transcript.to_dict()
        assert "metadata" in data
        assert "turns" in data
        assert "monitor_events" in data
        assert data["metadata"]["seed"] == 0
        assert data["metadata"]["actual_turns"] == 4  # 2 × 2

        # Should be JSON-serializable.
        json_str = json.dumps(data)
        assert len(json_str) > 100

    def test_transcript_has_timestamps(self, mania_config: LoomConfig) -> None:
        persona = MockModel(responses=["R."], name="mock/p")
        target = MockModel(responses=["R."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=1)

        assert transcript.started_at
        assert transcript.finished_at
        assert transcript.turns[0].timestamp

    def test_transcript_config_hash(self, mania_config: LoomConfig) -> None:
        persona = MockModel(responses=["R."], name="mock/p")
        target = MockModel(responses=["R."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=1)

        h = transcript.config_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_seed_produces_reproducible_scaffolding(
        self, mania_config: LoomConfig
    ) -> None:
        """Same seed → same injection schedule + intensities (not same model output)."""
        persona = MockModel(name="mock/p")
        target = MockModel(name="mock/t")

        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        t1 = runner.execute(n_turns=5, seed=42)
        # Reset mock counters.
        persona.call_count = 0
        target.call_count = 0
        t2 = runner.execute(n_turns=5, seed=42)

        # Injection types and intensities should be identical.
        for r1, r2 in zip(t1.turns, t2.turns):
            assert r1.injection_type == r2.injection_type
            assert r1.intensities == r2.intensities
            assert r1.phase == r2.phase


class TestRolloutMonitorIntegration:
    """Rollout with monitors that trigger."""

    def test_repetition_monitor_triggers_reprompt(
        self, mania_config: LoomConfig
    ) -> None:
        # Persona always says a banned phrase.
        persona = MockModel(
            responses=["Have you ever felt like nobody listens?"],
            name="mock/p",
        )
        target = MockModel(responses=["Tell me more."], name="mock/t")
        runner = RolloutRunner(
            config=mania_config,
            target_model=target,
            persona_model=persona,
        )
        transcript = runner.execute(n_turns=2)

        # Monitor should have triggered.
        events = transcript.monitor_events
        assert len(events) > 0
        assert any(e.monitor == "repetition" for e in events)

        # Retries should be recorded.
        persona_turns = [t for t in transcript.turns if t.role == "persona"]
        assert any(t.retries > 0 for t in persona_turns)
