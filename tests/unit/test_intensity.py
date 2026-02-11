"""Unit tests for intensity curve interpolation."""

from __future__ import annotations

import pytest

from loom.assembler.intensity import IntensityResolver, _curve_transform
from loom.schema.config import CurveParams, LoomConfig
from loom.schema.loader import load_config
from pathlib import Path


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "examples"


@pytest.fixture
def mania_config() -> LoomConfig:
    return load_config(CONFIGS_DIR / "mania_patient.yaml")


class TestCurveTransform:
    """Raw curve transform: progress [0,1] → transformed [0,1]."""

    def test_linear_at_zero(self) -> None:
        cp = CurveParams(type="linear")
        assert _curve_transform(cp, 0.0) == 0.0

    def test_linear_at_one(self) -> None:
        cp = CurveParams(type="linear")
        assert _curve_transform(cp, 1.0) == 1.0

    def test_linear_at_half(self) -> None:
        cp = CurveParams(type="linear")
        assert _curve_transform(cp, 0.5) == pytest.approx(0.5)

    def test_sigmoid_at_midpoint(self) -> None:
        cp = CurveParams(type="sigmoid", midpoint_pct=0.5)
        val = _curve_transform(cp, 0.5)
        # At midpoint, normalized sigmoid should be ~0.5.
        assert val == pytest.approx(0.5, abs=0.05)

    def test_sigmoid_at_zero(self) -> None:
        cp = CurveParams(type="sigmoid", midpoint_pct=0.5)
        val = _curve_transform(cp, 0.0)
        assert val == pytest.approx(0.0, abs=0.01)

    def test_sigmoid_at_one(self) -> None:
        cp = CurveParams(type="sigmoid", midpoint_pct=0.5)
        val = _curve_transform(cp, 1.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_sigmoid_monotonic(self) -> None:
        cp = CurveParams(type="sigmoid", midpoint_pct=0.5)
        values = [_curve_transform(cp, p / 10.0) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-9

    def test_delayed_ramp_before_delay(self) -> None:
        cp = CurveParams(type="delayed_ramp", delay_pct=0.6)
        assert _curve_transform(cp, 0.0) == 0.0
        assert _curve_transform(cp, 0.3) == 0.0
        assert _curve_transform(cp, 0.59) == 0.0

    def test_delayed_ramp_at_delay(self) -> None:
        cp = CurveParams(type="delayed_ramp", delay_pct=0.6)
        assert _curve_transform(cp, 0.6) == 0.0

    def test_delayed_ramp_after_delay(self) -> None:
        cp = CurveParams(type="delayed_ramp", delay_pct=0.6)
        val = _curve_transform(cp, 0.8)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_delayed_ramp_at_one(self) -> None:
        cp = CurveParams(type="delayed_ramp", delay_pct=0.6)
        assert _curve_transform(cp, 1.0) == pytest.approx(1.0, abs=0.01)

    def test_step_single_threshold(self) -> None:
        cp = CurveParams(type="step", step_thresholds=[0.5])
        assert _curve_transform(cp, 0.0) == 0.0
        assert _curve_transform(cp, 0.49) == 0.0
        assert _curve_transform(cp, 0.5) == 1.0
        assert _curve_transform(cp, 1.0) == 1.0

    def test_step_multiple_thresholds(self) -> None:
        cp = CurveParams(type="step", step_thresholds=[0.25, 0.5, 0.75])
        assert _curve_transform(cp, 0.0) == pytest.approx(0.0)
        assert _curve_transform(cp, 0.25) == pytest.approx(1 / 3, abs=0.01)
        assert _curve_transform(cp, 0.5) == pytest.approx(2 / 3, abs=0.01)
        assert _curve_transform(cp, 0.75) == pytest.approx(1.0, abs=0.01)


class TestIntensityResolver:
    """IntensityResolver: turn → {dimension: value}."""

    def test_resolve_turn_zero(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        result = resolver.resolve(0)
        assert "belief_intensity" in result
        assert "distress" in result
        assert "action_proximity" in result
        # Turn 0 should be near start values.
        assert result["belief_intensity"] == pytest.approx(0.1, abs=0.02)
        assert result["distress"] == pytest.approx(0.15, abs=0.02)
        assert result["action_proximity"] == pytest.approx(0.0, abs=0.02)

    def test_resolve_turn_last(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        result = resolver.resolve(49)  # Last turn of 50
        # Should be near end values.
        assert result["belief_intensity"] == pytest.approx(0.85, abs=0.02)
        assert result["distress"] == pytest.approx(0.75, abs=0.02)
        assert result["action_proximity"] == pytest.approx(0.7, abs=0.02)

    def test_resolve_halfway(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        result = resolver.resolve(24)  # ~50% through
        # Linear distress should be about halfway between 0.15 and 0.75.
        assert result["distress"] == pytest.approx(0.45, abs=0.05)
        # Sigmoid belief at midpoint should be ~halfway.
        assert result["belief_intensity"] == pytest.approx(0.475, abs=0.05)

    def test_values_respect_bounds(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        for turn in range(50):
            result = resolver.resolve(turn)
            for name, val in result.items():
                dim = mania_config.trajectory.dimensions[name]
                assert val >= dim.min_value - 0.001, f"{name} at turn {turn} below min"
                assert val <= dim.max_value + 0.001, f"{name} at turn {turn} above max"
                assert val <= mania_config.safety.intensity_ceiling + 0.001

    def test_resolve_phase_progression(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        phases_seen = []
        for turn in range(50):
            phase = resolver.resolve_phase(turn)
            if not phases_seen or phases_seen[-1] != phase:
                phases_seen.append(phase)
        # Should progress through rapport → disclosure → escalation → crisis → plateau.
        assert phases_seen == ["rapport", "disclosure", "escalation", "crisis", "plateau"]

    def test_resolve_phase_boundaries(self, mania_config: LoomConfig) -> None:
        resolver = IntensityResolver(mania_config)
        # rapport ends at 15% of 50 turns ≈ turn 7.
        # Turn 7 = progress 7/49 ≈ 0.143, which is < 0.15 → still rapport.
        # Turn 8 = progress 8/49 ≈ 0.163, which is > 0.15 → disclosure.
        assert resolver.resolve_phase(7) == "rapport"
        assert resolver.resolve_phase(8) == "disclosure"
