"""Stagnation monitor: detects agreement loops and conversational dead air.

Uses TF-IDF cosine similarity (no external embedding API dependency for v0,
per design review decision #5). Checks two things:

1. Self-similarity: are recent persona messages too similar to each other?
   (Indicates the persona is repeating itself.)
2. Convergence: are persona messages becoming too similar to interlocutor
   messages? (Indicates the persona is adopting the interlocutor's framing.)
"""

from __future__ import annotations

import math
import re
from collections import Counter

from loom.monitors.base import Monitor, MonitorAction, MonitorResult
from loom.schema.config import StagnationDetectionSpec


class StagnationMonitor(Monitor):
    """Detects agreement loops using TF-IDF cosine similarity."""

    def __init__(
        self,
        config: StagnationDetectionSpec,
        persona_name: str = "",
        next_revelation_fn: callable = None,  # type: ignore[type-arg]
    ) -> None:
        self._config = config
        self._persona_name = persona_name
        self._next_revelation_fn = next_revelation_fn or (lambda: "(new topic)")
        self._intervention_template = config.intervention_template

    @property
    def name(self) -> str:
        return "stagnation"

    def check(
        self,
        turn: int,
        response: str,
        history: list[dict[str, str]],
    ) -> MonitorResult:
        if not self._config.enabled:
            return MonitorResult(action=MonitorAction.OK)

        if turn < self._config.min_turn:
            return MonitorResult(action=MonitorAction.OK)

        # Collect recent persona messages from history.
        persona_msgs = _extract_role_messages(history, "persona")
        interlocutor_msgs = _extract_role_messages(history, "interlocutor")

        window = self._config.window
        recent_persona = persona_msgs[-(window - 1):] + [response]

        if len(recent_persona) < 2:
            return MonitorResult(action=MonitorAction.OK)

        # Check self-similarity.
        self_sim = _average_pairwise_similarity(recent_persona)
        if self_sim >= self._config.similarity_threshold:
            return self._trigger(
                f"Persona self-similarity {self_sim:.2f} >= {self._config.similarity_threshold}",
                {"self_similarity": round(self_sim, 3)},
            )

        # Check convergence with interlocutor.
        if interlocutor_msgs:
            recent_interlocutor = interlocutor_msgs[-window:]
            conv_sim = _cross_similarity(recent_persona, recent_interlocutor)
            if conv_sim >= self._config.convergence_threshold:
                return self._trigger(
                    f"Persona-interlocutor convergence {conv_sim:.2f} >= "
                    f"{self._config.convergence_threshold}",
                    {"convergence_similarity": round(conv_sim, 3)},
                )

        return MonitorResult(action=MonitorAction.OK)

    def _trigger(self, reason: str, details: dict) -> MonitorResult:
        injection_text = self._intervention_template
        if injection_text:
            next_rev = self._next_revelation_fn()
            injection_text = injection_text.replace("{name}", self._persona_name)
            injection_text = injection_text.replace("{next_unused_revelation}", next_rev)
        return MonitorResult(
            action=MonitorAction.EMERGENCY_INJECTION,
            reason=reason,
            injection_text=injection_text,
            details=details,
        )


def _extract_role_messages(
    history: list[dict[str, str]], role: str
) -> list[str]:
    """Extract message content for a given role from history."""
    return [m["content"] for m in history if m.get("role") == role]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _tf_vector(text: str) -> Counter:
    """Term frequency vector."""
    tokens = _tokenize(text)
    return Counter(tokens)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two term-frequency vectors."""
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _average_pairwise_similarity(texts: list[str]) -> float:
    """Average pairwise TF cosine similarity across a list of texts."""
    if len(texts) < 2:
        return 0.0
    vectors = [_tf_vector(t) for t in texts]
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += _cosine_similarity(vectors[i], vectors[j])
            count += 1
    return total / count if count else 0.0


def _cross_similarity(group_a: list[str], group_b: list[str]) -> float:
    """Average cross-group TF cosine similarity."""
    if not group_a or not group_b:
        return 0.0
    vectors_a = [_tf_vector(t) for t in group_a]
    vectors_b = [_tf_vector(t) for t in group_b]
    total = 0.0
    count = 0
    for va in vectors_a:
        for vb in vectors_b:
            total += _cosine_similarity(va, vb)
            count += 1
    return total / count if count else 0.0
