"""Intensity interpolation: turn number → per-dimension intensity values.

Each DimensionSpec defines a curve (sigmoid, linear, delayed_ramp, step)
that maps conversation progress (0→1) to an intensity value between
start_value and end_value, clamped to [min_value, max_value].
"""

from __future__ import annotations

import math

from loom.schema.config import LoomConfig, DimensionSpec, CurveParams
from loom.schema.defaults import CurveType


class IntensityResolver:
    """Resolves per-dimension intensity values for a given turn."""

    def __init__(self, config: LoomConfig) -> None:
        self._config = config
        self._expected_turns = config.trajectory.expected_turns or 1
        self._ceiling = config.safety.intensity_ceiling

    def resolve(self, turn: int) -> dict[str, float]:
        """Return {dimension_name: intensity_value} for the given turn.

        Turn is 0-indexed. Progress = turn / (expected_turns - 1), clamped
        to [0, 1]. For a 50-turn conversation, turn 0 → progress 0.0,
        turn 49 → progress 1.0.
        """
        if self._expected_turns <= 1:
            progress = 1.0
        else:
            progress = min(turn / (self._expected_turns - 1), 1.0)
            progress = max(progress, 0.0)

        result: dict[str, float] = {}
        for name, dim in self._config.trajectory.dimensions.items():
            raw = _interpolate(dim.curve, dim.start_value, dim.end_value, progress)
            clamped = max(dim.min_value, min(raw, dim.max_value))
            clamped = min(clamped, self._ceiling)
            result[name] = round(clamped, 4)
        return result

    def resolve_phase(self, turn: int) -> str:
        """Return the name of the active phase at the given turn."""
        if self._expected_turns <= 1:
            progress = 1.0
        else:
            progress = min(turn / (self._expected_turns - 1), 1.0)

        for phase in self._config.trajectory.phases:
            ec = phase.end_condition
            # For pct-based end conditions, the phase is active while
            # progress < end_condition.value (inclusive for the last phase).
            if ec.type == "pct" and progress <= ec.value:
                return phase.name

        # Fallback: return last phase.
        return self._config.trajectory.phases[-1].name


def _interpolate(
    curve: CurveParams,
    start: float,
    end: float,
    progress: float,
) -> float:
    """Map progress [0,1] to a value between start and end using the curve."""
    t = _curve_transform(curve, progress)
    return start + t * (end - start)


def _curve_transform(curve: CurveParams, progress: float) -> float:
    """Apply the curve function to progress, returning a [0,1] transformed value.

    - linear: t = progress
    - sigmoid: logistic function centered at midpoint_pct
    - delayed_ramp: flat near 0 until delay_pct, then linear ramp
    - step: discrete jumps at thresholds
    """
    if curve.type == CurveType.LINEAR:
        return progress

    if curve.type == CurveType.SIGMOID:
        midpoint = curve.midpoint_pct or 0.5
        # Steepness chosen so the curve covers ~95% of range within
        # the conversation. k=12 gives a nice S-shape.
        k = 12.0
        raw = 1.0 / (1.0 + math.exp(-k * (progress - midpoint)))
        # Normalize so that raw(0)→0 and raw(1)→1.
        raw_at_0 = 1.0 / (1.0 + math.exp(-k * (0.0 - midpoint)))
        raw_at_1 = 1.0 / (1.0 + math.exp(-k * (1.0 - midpoint)))
        if raw_at_1 == raw_at_0:
            return progress  # degenerate case
        return (raw - raw_at_0) / (raw_at_1 - raw_at_0)

    if curve.type == CurveType.DELAYED_RAMP:
        delay = curve.delay_pct or 0.5
        if progress <= delay:
            return 0.0
        # Linear ramp from delay_pct to 1.0.
        return (progress - delay) / (1.0 - delay)

    if curve.type == CurveType.STEP:
        thresholds = curve.step_thresholds or [0.5]
        # Count how many thresholds have been passed.
        n_steps = len(thresholds) + 1  # number of levels
        passed = sum(1 for th in thresholds if progress >= th)
        return passed / (n_steps - 1) if n_steps > 1 else 1.0

    # Unknown curve type — treat as linear.
    return progress
