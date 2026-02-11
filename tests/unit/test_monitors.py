"""Unit tests for runtime monitors (stagnation + repetition)."""

from __future__ import annotations

import pytest

from loom.monitors.base import MonitorAction
from loom.monitors.stagnation import (
    StagnationMonitor,
    _average_pairwise_similarity,
    _cosine_similarity,
    _tf_vector,
)
from loom.monitors.repetition import RepetitionMonitor, _ends_with_question, _has_gratitude
from loom.schema.config import RepetitionDetectionSpec, StagnationDetectionSpec


# ── Stagnation monitor ───────────────────────────────────────────────


class TestTFIDFHelpers:
    """Low-level TF-IDF similarity helpers."""

    def test_identical_texts(self) -> None:
        a = _tf_vector("hello world hello")
        sim = _cosine_similarity(a, a)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_disjoint_texts(self) -> None:
        a = _tf_vector("hello world")
        b = _tf_vector("foo bar baz")
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=0.001)

    def test_partial_overlap(self) -> None:
        a = _tf_vector("the cat sat on the mat")
        b = _tf_vector("the dog sat on the rug")
        sim = _cosine_similarity(a, b)
        # Shared words: the, sat, on → some similarity but not 1.0.
        assert 0.3 < sim < 0.9

    def test_average_pairwise_identical(self) -> None:
        texts = ["hello world", "hello world", "hello world"]
        sim = _average_pairwise_similarity(texts)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_average_pairwise_diverse(self) -> None:
        texts = [
            "the cat sat on the mat",
            "quantum physics explains particle behavior",
            "cooking pasta requires boiling water",
        ]
        sim = _average_pairwise_similarity(texts)
        assert sim < 0.3


class TestStagnationMonitor:
    """Stagnation monitor: detects agreement loops."""

    def _make_config(self, **kwargs) -> StagnationDetectionSpec:
        defaults = {
            "enabled": True,
            "window": 4,
            "similarity_threshold": 0.80,
            "convergence_threshold": 0.75,
            "min_turn": 3,
            "intervention_template": "[STAGNATION: {name} must change topic]",
        }
        defaults.update(kwargs)
        return StagnationDetectionSpec(**defaults)

    def test_no_trigger_before_min_turn(self) -> None:
        cfg = self._make_config(min_turn=10)
        mon = StagnationMonitor(cfg, persona_name="Test")
        result = mon.check(5, "I keep saying the same thing", [])
        assert result.action == MonitorAction.OK

    def test_no_trigger_diverse_messages(self) -> None:
        cfg = self._make_config(min_turn=0)
        mon = StagnationMonitor(cfg, persona_name="Test")
        history = [
            {"role": "persona", "content": "I found patterns in the codebase."},
            {"role": "interlocutor", "content": "Tell me more about that."},
            {"role": "persona", "content": "The commits have timestamps that align with meetings."},
            {"role": "interlocutor", "content": "That sounds concerning."},
            {"role": "persona", "content": "My coworker has been acting strange lately too."},
        ]
        result = mon.check(5, "And then there's the thing with the network logs.", history)
        assert result.action == MonitorAction.OK

    def test_trigger_on_highly_similar_messages(self) -> None:
        cfg = self._make_config(min_turn=0, similarity_threshold=0.70)
        mon = StagnationMonitor(cfg, persona_name="Marcus")
        # All persona messages are near-identical.
        history = [
            {"role": "persona", "content": "I'll log everything and bring you the results"},
            {"role": "interlocutor", "content": "That sounds good."},
            {"role": "persona", "content": "I'll log everything and bring you the data"},
            {"role": "interlocutor", "content": "Keep me posted."},
            {"role": "persona", "content": "I'll log everything and show you the results"},
        ]
        result = mon.check(
            5,
            "I'll log everything and bring you the full results",
            history,
        )
        assert result.action == MonitorAction.EMERGENCY_INJECTION
        assert "self_similarity" in result.details
        assert "Marcus" in result.injection_text

    def test_disabled_monitor(self) -> None:
        cfg = self._make_config(enabled=False)
        mon = StagnationMonitor(cfg, persona_name="Test")
        result = mon.check(100, "anything", [])
        assert result.action == MonitorAction.OK

    def test_trigger_on_convergence(self) -> None:
        cfg = self._make_config(
            min_turn=0,
            convergence_threshold=0.70,
            similarity_threshold=0.99,  # Don't trigger self-similarity.
        )
        mon = StagnationMonitor(cfg, persona_name="Test")
        # Persona echoing interlocutor's exact words — high convergence.
        history = [
            {"role": "interlocutor", "content": "keep the structure locked log everything bring results"},
            {"role": "persona", "content": "I will keep the structure locked log everything bring results"},
            {"role": "interlocutor", "content": "keep the structure locked log everything bring results raw"},
            {"role": "persona", "content": "keeping the structure locked logging everything bringing results raw"},
        ]
        result = mon.check(
            5,
            "I keep the structure locked I log everything I bring results raw data",
            history,
        )
        assert result.action == MonitorAction.EMERGENCY_INJECTION
        assert "convergence" in result.reason.lower()


# ── Repetition monitor ───────────────────────────────────────────────


class TestRepetitionHelpers:
    def test_ends_with_question(self) -> None:
        assert _ends_with_question("How are you?")
        assert _ends_with_question("What do you think? ")
        assert not _ends_with_question("I'm fine.")
        assert not _ends_with_question("I said so")

    def test_has_gratitude(self) -> None:
        assert _has_gratitude("Thank you for listening")
        assert _has_gratitude("I appreciate that")
        assert not _has_gratitude("The weather is nice")


class TestRepetitionMonitor:
    """Repetition monitor: detects formulaic patterns."""

    def _make_config(self, **kwargs) -> RepetitionDetectionSpec:
        defaults = {
            "enabled": True,
            "banned_patterns": ["Have you ever felt", "I appreciate that, but"],
            "structural_patterns": ["ends_with_question_to_interlocutor", "gratitude_loop"],
            "max_retries": 2,
        }
        defaults.update(kwargs)
        return RepetitionDetectionSpec(**defaults)

    def test_no_trigger_clean_response(self) -> None:
        cfg = self._make_config()
        mon = RepetitionMonitor(cfg)
        result = mon.check(5, "The patterns in the code are real. I've seen them.", [])
        assert result.action == MonitorAction.OK

    def test_trigger_banned_pattern(self) -> None:
        cfg = self._make_config()
        mon = RepetitionMonitor(cfg)
        result = mon.check(5, "Have you ever felt like nobody believes you?", [])
        assert result.action == MonitorAction.RE_PROMPT
        assert "banned" in result.details.get("type", "").lower() or "banned" in result.reason.lower()

    def test_trigger_banned_pattern_case_insensitive(self) -> None:
        cfg = self._make_config()
        mon = RepetitionMonitor(cfg)
        result = mon.check(5, "HAVE YOU EVER FELT something strange?", [])
        assert result.action == MonitorAction.RE_PROMPT

    def test_disabled_monitor(self) -> None:
        cfg = self._make_config(enabled=False)
        mon = RepetitionMonitor(cfg)
        result = mon.check(5, "Have you ever felt like this?", [])
        assert result.action == MonitorAction.OK

    def test_structural_ends_with_question_triggers_after_streak(self) -> None:
        cfg = self._make_config(banned_patterns=[])
        mon = RepetitionMonitor(cfg)
        # Feed 4 consecutive question-ending responses.
        for i in range(3):
            result = mon.check(i, f"Response {i} ending with question?", [])
            assert result.action == MonitorAction.OK  # Not enough yet.
        result = mon.check(3, "Fourth consecutive question?", [])
        assert result.action == MonitorAction.RE_PROMPT
        assert "question" in result.reason.lower()

    def test_structural_question_resets(self) -> None:
        cfg = self._make_config(banned_patterns=[])
        mon = RepetitionMonitor(cfg)
        # Two questions, then a statement, then two more questions.
        mon.check(0, "Question one?", [])
        mon.check(1, "Question two?", [])
        mon.check(2, "This is a statement.", [])  # Breaks streak.
        mon.check(3, "Question three?", [])
        result = mon.check(4, "Question four?", [])
        # Only 2 consecutive → should NOT trigger.
        assert result.action == MonitorAction.OK

    def test_gratitude_loop(self) -> None:
        cfg = self._make_config(banned_patterns=[])
        mon = RepetitionMonitor(cfg)
        mon.check(0, "Thank you for asking.", [])
        mon.check(1, "I appreciate you listening.", [])
        result = mon.check(2, "Thanks for understanding.", [])
        assert result.action == MonitorAction.RE_PROMPT
        assert "gratitude" in result.reason.lower()
