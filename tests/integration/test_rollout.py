"""Integration tests for rollout runner with real APIs.

These tests make REAL API calls and cost money. They are:
- Marked with @pytest.mark.live and skipped by default.
- Run manually with: pytest --live tests/integration/
- Budget target: < €3 total for a full run.
- Kept short: 3 turns, 1 rollout.

API keys required as environment variables:
- OPENAI_API_KEY for OpenAI models
- ANTHROPIC_API_KEY for Anthropic models
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from loom.runner.rollout import RolloutRunner
from loom.schema.loader import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "examples"

# Skip all tests in this module unless --live is passed.
pytestmark = pytest.mark.live


@pytest.fixture
def student_config():
    """Student persona — cheapest config (2 dimensions, 3 phases, short responses)."""
    return load_config(CONFIGS_DIR / "student_persona.yaml")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIRollout:
    """Integration test with OpenAI models."""

    def test_short_rollout(self, student_config, tmp_path: Path) -> None:
        """3-turn rollout with gpt-4.1-mini. ~$0.01 estimated cost."""
        runner = RolloutRunner(
            config=student_config,
            # Use mini models to minimize cost.
            target_model="openai/gpt-4.1-mini",
            persona_model="openai/gpt-4.1-mini",
            seed=42,
        )
        transcript = runner.execute(n_turns=3, seed=42)

        # Basic structure checks.
        assert len(transcript.turns) == 6  # 3 persona + 3 interlocutor
        assert transcript.started_at
        assert transcript.finished_at

        # All turns have content.
        for turn in transcript.turns:
            assert len(turn.content) > 0, f"Empty content at turn {turn.turn} ({turn.role})"

        # Persona turns have intensities and phase.
        persona_turns = [t for t in transcript.turns if t.role == "persona"]
        for pt in persona_turns:
            assert pt.phase in {"guarded", "cracking", "engaging"}
            assert "confidence" in pt.intensities

        # Serialization works.
        data = transcript.to_dict()
        out_path = tmp_path / "test_rollout.json"
        out_path.write_text(json.dumps(data, indent=2))
        assert out_path.stat().st_size > 500

        # Log cost estimate.
        # gpt-4.1-mini: ~$0.15/1M input, $0.60/1M output.
        # 3 turns × ~500 tokens input × 2 models = ~3000 input tokens.
        # 3 turns × ~100 tokens output × 2 models = ~600 output tokens.
        # Estimated cost: ~$0.001
        print(f"\nTranscript saved to {out_path}")
        print(f"Turns: {len(transcript.turns)}")
        print(f"Monitor events: {len(transcript.monitor_events)}")


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicRollout:
    """Integration test with Anthropic models."""

    def test_short_rollout(self, student_config, tmp_path: Path) -> None:
        """3-turn rollout with claude-haiku. ~$0.01 estimated cost."""
        runner = RolloutRunner(
            config=student_config,
            target_model="anthropic/claude-haiku-4-5-20251001",
            persona_model="anthropic/claude-haiku-4-5-20251001",
            seed=42,
        )
        transcript = runner.execute(n_turns=3, seed=42)

        assert len(transcript.turns) == 6
        for turn in transcript.turns:
            assert len(turn.content) > 0

        data = transcript.to_dict()
        out_path = tmp_path / "test_rollout_anthropic.json"
        out_path.write_text(json.dumps(data, indent=2))
        assert out_path.stat().st_size > 500

        print(f"\nTranscript saved to {out_path}")
        print(f"Turns: {len(transcript.turns)}")
        print(f"Monitor events: {len(transcript.monitor_events)}")


@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")),
    reason="Both OPENAI_API_KEY and ANTHROPIC_API_KEY required",
)
class TestCrossProviderRollout:
    """Integration test with persona on one provider, interlocutor on another."""

    def test_openai_persona_anthropic_interlocutor(
        self, student_config, tmp_path: Path
    ) -> None:
        """Cross-provider rollout. ~$0.02 estimated cost."""
        runner = RolloutRunner(
            config=student_config,
            target_model="anthropic/claude-haiku-4-5-20251001",
            persona_model="openai/gpt-4.1-mini",
            seed=42,
        )
        transcript = runner.execute(n_turns=3, seed=42)

        assert len(transcript.turns) == 6
        assert transcript.persona_model == "gpt-4.1-mini"
        assert transcript.target_model == "claude-haiku-4-5-20251001"

        for turn in transcript.turns:
            assert len(turn.content) > 0

        data = transcript.to_dict()
        out_path = tmp_path / "test_rollout_cross.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nCross-provider transcript saved to {out_path}")
