"""Unit tests for Loom config schema validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from loom.schema.config import (
    CurveParams,
    DimensionSpec,
    LoomConfig,
    PhaseSpec,
    RevelationSpec,
    TrajectorySpec,
)
from loom.schema.loader import ConfigError, load_config


# ── Valid config parsing ─────────────────────────────────────────────


class TestValidConfigs:
    """Valid configs should parse without error."""

    def test_minimal_config(self, minimal_config_dict: dict) -> None:
        cfg = LoomConfig.model_validate(minimal_config_dict)
        assert cfg.persona.identity.name == "Test Person"
        assert cfg.trajectory.mode == "fixed_length"
        assert cfg.trajectory.expected_turns == 20
        assert len(cfg.trajectory.dimensions) == 1
        assert len(cfg.trajectory.phases) == 1

    def test_mania_config_from_file(self, mania_config_path: Path) -> None:
        cfg = load_config(mania_config_path)
        assert cfg.persona.identity.name == "Marcus Chen"
        assert cfg.trajectory.expected_turns == 50
        assert len(cfg.trajectory.dimensions) == 3
        assert len(cfg.trajectory.phases) == 5

    def test_all_example_configs_valid(self, configs_dir: Path) -> None:
        for yaml_file in sorted(configs_dir.glob("*.yaml")):
            cfg = load_config(yaml_file)
            assert cfg.schema_version == "0.1.0", f"Failed on {yaml_file.name}"

    def test_defaults_applied(self, minimal_config_dict: dict) -> None:
        cfg = LoomConfig.model_validate(minimal_config_dict)
        # Evaluation and safety get defaults.
        assert cfg.evaluation.adversary.type == "none"
        assert cfg.safety.intensity_ceiling == 0.9
        dim = cfg.trajectory.dimensions["intensity"]
        assert dim.min_value == 0.0
        assert dim.max_value == 0.9

    def test_revelation_spec_parsing(self) -> None:
        rev = RevelationSpec(
            topic="test_topic",
            variants={"subtle": "hint hint", "direct": "here it is"},
        )
        assert rev.topic == "test_topic"
        assert "subtle" in rev.variants


# ── Missing required fields ──────────────────────────────────────────


class TestMissingFields:
    """Missing required fields should raise ValidationError."""

    def test_missing_persona(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["persona"]
        with pytest.raises(ValidationError, match="persona"):
            LoomConfig.model_validate(d)

    def test_missing_trajectory(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["trajectory"]
        with pytest.raises(ValidationError, match="trajectory"):
            LoomConfig.model_validate(d)

    def test_missing_interaction(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["interaction"]
        with pytest.raises(ValidationError, match="interaction"):
            LoomConfig.model_validate(d)

    def test_missing_identity_name(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["persona"]["identity"]["name"]
        with pytest.raises(ValidationError, match="name"):
            LoomConfig.model_validate(d)

    def test_missing_schema_version(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["schema_version"]
        with pytest.raises(ValidationError, match="schema_version"):
            LoomConfig.model_validate(d)

    def test_missing_expected_turns_fixed_length(
        self, minimal_config_dict: dict
    ) -> None:
        d = copy.deepcopy(minimal_config_dict)
        del d["trajectory"]["expected_turns"]
        with pytest.raises(ValidationError, match="expected_turns"):
            LoomConfig.model_validate(d)


# ── Invalid enum values ──────────────────────────────────────────────


class TestInvalidEnums:
    """Invalid enum values should be caught."""

    def test_invalid_trajectory_mode(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["mode"] = "invalid_mode"
        with pytest.raises(ValidationError, match="mode"):
            LoomConfig.model_validate(d)

    def test_invalid_curve_type(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["dimensions"]["intensity"]["curve"]["type"] = "cubic"
        with pytest.raises(ValidationError):
            LoomConfig.model_validate(d)

    def test_invalid_resistance_level(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["interaction"]["anti_capitulation"]["resistance_level"] = "extreme"
        with pytest.raises(ValidationError):
            LoomConfig.model_validate(d)

    def test_invalid_adversary_type(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["evaluation"] = {"adversary": {"type": "nuclear"}}
        with pytest.raises(ValidationError):
            LoomConfig.model_validate(d)


# ── Phase coverage validation ────────────────────────────────────────


class TestPhaseCoverage:
    """Phase end conditions must cover 100% of the conversation."""

    def test_last_phase_not_1_0(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["phases"][0]["end_condition"]["value"] = 0.8
        with pytest.raises(ValidationError, match="must end at 1.0"):
            LoomConfig.model_validate(d)

    def test_phases_not_ascending(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["phases"] = [
            {
                "name": "phase_a",
                "end_condition": {"type": "pct", "value": 0.5},
                "requirements": ["a"],
            },
            {
                "name": "phase_b",
                "end_condition": {"type": "pct", "value": 0.3},
                "requirements": ["b"],
            },
            {
                "name": "phase_c",
                "end_condition": {"type": "pct", "value": 1.0},
                "requirements": ["c"],
            },
        ]
        with pytest.raises(ValidationError, match="must be greater than"):
            LoomConfig.model_validate(d)

    def test_multi_phase_valid(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["phases"] = [
            {
                "name": "early",
                "end_condition": {"type": "pct", "value": 0.4},
                "requirements": ["a"],
            },
            {
                "name": "late",
                "end_condition": {"type": "pct", "value": 1.0},
                "requirements": ["b"],
            },
        ]
        cfg = LoomConfig.model_validate(d)
        assert len(cfg.trajectory.phases) == 2

    def test_empty_phases_rejected(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["trajectory"]["phases"] = []
        with pytest.raises(ValidationError, match="at least one phase"):
            LoomConfig.model_validate(d)


# ── Dimension bounds ─────────────────────────────────────────────────


class TestDimensionBounds:
    """Dimension values must respect min < max, and start/end within bounds."""

    def test_min_exceeds_max(self) -> None:
        with pytest.raises(ValidationError, match="min_value.*must be less than"):
            DimensionSpec(
                description="test",
                levels={"low": "a", "high": "b"},
                curve=CurveParams(type="linear"),
                start_value=0.1,
                end_value=0.8,
                min_value=0.9,
                max_value=0.5,
            )

    def test_start_below_min(self) -> None:
        with pytest.raises(ValidationError, match="start_value.*must be between"):
            DimensionSpec(
                description="test",
                levels={"low": "a", "high": "b"},
                curve=CurveParams(type="linear"),
                start_value=0.0,
                end_value=0.7,
                min_value=0.1,
                max_value=0.9,
            )

    def test_end_above_max(self) -> None:
        with pytest.raises(ValidationError, match="end_value.*must be between"):
            DimensionSpec(
                description="test",
                levels={"low": "a", "high": "b"},
                curve=CurveParams(type="linear"),
                start_value=0.1,
                end_value=0.95,
                min_value=0.0,
                max_value=0.9,
            )


# ── Intensity ceiling ────────────────────────────────────────────────


class TestIntensityCeiling:
    """Dimension max_values must not exceed safety intensity_ceiling."""

    def test_dim_exceeds_ceiling(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["safety"] = {"intensity_ceiling": 0.7}
        # Dimension's default max_value is 0.9, which exceeds 0.7.
        with pytest.raises(ValidationError, match="exceeds.*intensity_ceiling"):
            LoomConfig.model_validate(d)

    def test_dim_within_ceiling(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["safety"] = {"intensity_ceiling": 0.95}
        cfg = LoomConfig.model_validate(d)
        assert cfg.safety.intensity_ceiling == 0.95


# ── Schema version ───────────────────────────────────────────────────


class TestSchemaVersion:
    """schema_version must be a recognized version string."""

    def test_unsupported_version(self, minimal_config_dict: dict) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["schema_version"] = "2.0.0"
        with pytest.raises(ValidationError, match="Unsupported schema_version"):
            LoomConfig.model_validate(d)


# ── Curve parameter validation ───────────────────────────────────────


class TestCurveParams:
    """Curve types must have their required parameters."""

    def test_sigmoid_requires_midpoint(self) -> None:
        with pytest.raises(ValidationError, match="midpoint_pct"):
            CurveParams(type="sigmoid")

    def test_delayed_ramp_requires_delay(self) -> None:
        with pytest.raises(ValidationError, match="delay_pct"):
            CurveParams(type="delayed_ramp")

    def test_step_requires_thresholds(self) -> None:
        with pytest.raises(ValidationError, match="step_thresholds"):
            CurveParams(type="step")

    def test_linear_no_extra_params(self) -> None:
        cp = CurveParams(type="linear")
        assert cp.type == "linear"

    def test_sigmoid_valid(self) -> None:
        cp = CurveParams(type="sigmoid", midpoint_pct=0.5)
        assert cp.midpoint_pct == 0.5

    def test_step_valid(self) -> None:
        cp = CurveParams(type="step", step_thresholds=[0.25, 0.5, 0.75])
        assert len(cp.step_thresholds) == 3


# ── Loader edge cases ────────────────────────────────────────────────


class TestLoader:
    """Config loader should handle file errors gracefully."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_wrong_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(ConfigError, match="Expected .yaml"):
            load_config(f)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("{{{{")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(f)

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="Expected a YAML mapping"):
            load_config(f)

    def test_validation_error_readable(self, tmp_path: Path) -> None:
        f = tmp_path / "partial.yaml"
        f.write_text('schema_version: "0.1.0"\n')
        with pytest.raises(ConfigError, match="Validation errors"):
            load_config(f)


# ── Response length phase reference ──────────────────────────────────


class TestResponseLengthPhaseRef:
    """response_length.by_phase must only reference existing phases."""

    def test_unknown_phase_in_response_length(
        self, minimal_config_dict: dict
    ) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["interaction"]["response_length"]["by_phase"] = {
            "nonexistent_phase": "20-40 words"
        }
        with pytest.raises(ValidationError, match="unknown phase"):
            LoomConfig.model_validate(d)

    def test_valid_phase_in_response_length(
        self, minimal_config_dict: dict
    ) -> None:
        d = copy.deepcopy(minimal_config_dict)
        d["interaction"]["response_length"]["by_phase"] = {
            "only_phase": "20-40 words"
        }
        cfg = LoomConfig.model_validate(d)
        assert "only_phase" in cfg.interaction.response_length.by_phase
