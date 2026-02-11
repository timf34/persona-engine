"""Transcript data model and JSON serialization.

A Transcript represents a single rollout: the full message history plus
per-turn metadata (prescribed intensities, phase, injection type, monitor
events) and run-level metadata (config hash, seed, timestamps, models).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loom.assembler.templates import TEMPLATE_VERSION
from loom.schema.defaults import SCHEMA_VERSION


@dataclass
class MonitorEvent:
    """A runtime monitor trigger event."""

    turn: int
    monitor: str  # "stagnation" or "repetition"
    reason: str
    action: str  # "emergency_injection", "re_prompt", "logged_only"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "monitor": self.monitor,
            "reason": self.reason,
            "action": self.action,
            "details": self.details,
        }


@dataclass
class TurnRecord:
    """Per-turn data stored in the transcript."""

    turn: int
    role: str  # "persona" or "interlocutor"
    content: str
    phase: str
    intensities: dict[str, float]
    injection_type: str
    monitor_events: list[MonitorEvent] = field(default_factory=list)
    retries: int = 0  # how many times this turn was re-prompted
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "phase": self.phase,
            "intensities": self.intensities,
            "injection_type": self.injection_type,
        }
        if self.monitor_events:
            d["monitor_events"] = [e.to_dict() for e in self.monitor_events]
        if self.retries > 0:
            d["retries"] = self.retries
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


@dataclass
class Transcript:
    """Complete transcript for a single rollout."""

    config_yaml: str  # raw YAML content for reproducibility
    persona_model: str
    target_model: str
    seed: int
    n_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    monitor_events: list[MonitorEvent] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def add_turn(self, record: TurnRecord) -> None:
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()
        self.turns.append(record)

    def add_monitor_event(self, event: MonitorEvent) -> None:
        self.monitor_events.append(event)

    def config_hash(self) -> str:
        return hashlib.sha256(self.config_yaml.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "schema_version": SCHEMA_VERSION,
                "template_version": TEMPLATE_VERSION,
                "config_hash": self.config_hash(),
                "persona_model": self.persona_model,
                "target_model": self.target_model,
                "seed": self.seed,
                "n_turns": self.n_turns,
                "actual_turns": len(self.turns),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            },
            "turns": [t.to_dict() for t in self.turns],
            "monitor_events": [e.to_dict() for e in self.monitor_events],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
