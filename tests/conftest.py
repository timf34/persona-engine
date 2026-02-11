"""Shared test fixtures for Loom tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so integration tests can find API keys.
load_dotenv()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "examples"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that make real API calls (marked @pytest.mark.live).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="need --live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def mania_config_path() -> Path:
    return CONFIGS_DIR / "mania_patient.yaml"


@pytest.fixture
def minimal_config_dict() -> dict:
    """Smallest valid config as a Python dict (for Pydantic model_validate)."""
    return {
        "schema_version": "0.1.0",
        "persona": {
            "identity": {
                "name": "Test Person",
                "age": 30,
                "background": "A test persona.",
                "backstory_summary": "Created for testing.",
            },
            "capability_bounds": {
                "knowledge_ceiling": "general knowledge",
                "vocabulary_level": "conversational",
                "reasoning_style": "logical",
            },
            "cognitive_style": "straightforward",
            "speech_patterns": ["speaks normally"],
            "recovery_behavior": "acknowledges the error",
            "emotional_responses": {"challenged": "stays calm"},
        },
        "trajectory": {
            "mode": "fixed_length",
            "expected_turns": 20,
            "dimensions": {
                "intensity": {
                    "description": "General intensity",
                    "levels": {"low": "calm", "high": "intense"},
                    "curve": {"type": "linear"},
                    "start_value": 0.1,
                    "end_value": 0.8,
                }
            },
            "phases": [
                {
                    "name": "only_phase",
                    "end_condition": {"type": "pct", "value": 1.0},
                    "requirements": ["be present"],
                }
            ],
        },
        "interaction": {
            "injection": {
                "frequency": 5,
                "reminder_frequency": 2,
                "reminder_template": "[REMINDER: stay in character]",
            },
            "anti_capitulation": {
                "resistance_level": "low",
            },
            "response_length": {
                "default": "30-80 words",
            },
        },
    }
