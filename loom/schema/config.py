"""Pydantic v2 models for the Loom YAML config schema."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from loom.schema.defaults import (
    SUPPORTED_SCHEMA_VERSIONS,
    AdversaryType,
    CurveType,
    EndConditionType,
    ResistanceLevel,
    TrajectoryMode,
)


# ── Persona ──────────────────────────────────────────────────────────


class IdentitySpec(BaseModel):
    """Who the persona is."""

    name: str
    age: int
    background: str
    backstory_summary: str


class BoundsSpec(BaseModel):
    """Knowledge and capability limits the persona must respect."""

    knowledge_ceiling: str
    vocabulary_level: str
    reasoning_style: str


class PersonaSpec(BaseModel):
    """Complete persona definition."""

    identity: IdentitySpec
    capability_bounds: BoundsSpec
    cognitive_style: str
    speech_patterns: list[str]
    recovery_behavior: str
    emotional_responses: dict[str, str]


# ── Trajectory ───────────────────────────────────────────────────────


class CurveParams(BaseModel):
    """Curve definition with type-specific parameters."""

    type: CurveType
    midpoint_pct: float | None = Field(
        default=None,
        description="Sigmoid inflection point as fraction of conversation (0-1).",
    )
    delay_pct: float | None = Field(
        default=None,
        description="Delayed ramp: stay near start_value until this fraction.",
    )
    step_thresholds: list[float] | None = Field(
        default=None,
        description="Step curve: fractions of conversation where jumps occur.",
    )

    @model_validator(mode="after")
    def check_curve_params(self) -> CurveParams:
        if self.type == CurveType.SIGMOID and self.midpoint_pct is None:
            raise ValueError("sigmoid curve requires midpoint_pct")
        if self.type == CurveType.DELAYED_RAMP and self.delay_pct is None:
            raise ValueError("delayed_ramp curve requires delay_pct")
        if self.type == CurveType.STEP and not self.step_thresholds:
            raise ValueError("step curve requires step_thresholds")
        return self


class DimensionSpec(BaseModel):
    """A single trajectory dimension (e.g. belief_intensity, distress).

    `start_value`/`end_value` are trajectory endpoints — where the curve
    begins and ends.  `min_value`/`max_value` are safety bounds that clamp
    the interpolated value at every turn.
    """

    description: str
    levels: dict[str, str]
    curve: CurveParams
    start_value: float = Field(ge=0.0, le=1.0)
    end_value: float = Field(ge=0.0, le=1.0)
    min_value: float = Field(default=0.0, ge=0.0, le=1.0)
    max_value: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(self) -> DimensionSpec:
        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be less than "
                f"max_value ({self.max_value})"
            )
        if not (self.min_value <= self.start_value <= self.max_value):
            raise ValueError(
                f"start_value ({self.start_value}) must be between "
                f"min_value ({self.min_value}) and max_value ({self.max_value})"
            )
        if not (self.min_value <= self.end_value <= self.max_value):
            raise ValueError(
                f"end_value ({self.end_value}) must be between "
                f"min_value ({self.min_value}) and max_value ({self.max_value})"
            )
        return self


class EndCondition(BaseModel):
    """When a phase ends — pct of conversation or absolute turn number."""

    type: EndConditionType
    value: float = Field(
        description="For pct: fraction 0-1. For turn: turn number."
    )


class RevelationSpec(BaseModel):
    """A topic the persona can reveal, with intensity-keyed variants."""

    topic: str
    variants: dict[str, str] = Field(
        description="Intensity label (e.g. subtle/moderate/direct) → text."
    )


class PhaseSpec(BaseModel):
    """A named phase of the conversation with requirements and boundaries."""

    name: str
    end_condition: EndCondition
    requirements: list[str]
    forbidden: list[str] = Field(default_factory=list)
    revelations: list[RevelationSpec] = Field(default_factory=list)


class TrajectorySpec(BaseModel):
    """Conversation arc: dimensions, phases, and overall structure."""

    mode: TrajectoryMode
    expected_turns: int | None = None
    dimensions: dict[str, DimensionSpec]
    phases: list[PhaseSpec]

    @model_validator(mode="after")
    def check_trajectory(self) -> TrajectorySpec:
        if self.mode == TrajectoryMode.FIXED_LENGTH and self.expected_turns is None:
            raise ValueError("fixed_length mode requires expected_turns")
        if self.expected_turns is not None and self.expected_turns <= 0:
            raise ValueError("expected_turns must be positive")
        if not self.phases:
            raise ValueError("at least one phase is required")
        if not self.dimensions:
            raise ValueError("at least one dimension is required")
        # Validate phase ordering and coverage.
        self._check_phase_coverage()
        return self

    def _check_phase_coverage(self) -> None:
        """Phases must be in ascending order and the last must end at 1.0 (pct)."""
        prev_value = 0.0
        for phase in self.phases:
            ec = phase.end_condition
            if ec.type == EndConditionType.PCT:
                if ec.value <= prev_value:
                    raise ValueError(
                        f"Phase '{phase.name}' end_condition ({ec.value}) "
                        f"must be greater than previous ({prev_value})"
                    )
                prev_value = ec.value
        # Last phase must reach 1.0 for pct-based conditions.
        last = self.phases[-1].end_condition
        if last.type == EndConditionType.PCT and last.value != 1.0:
            raise ValueError(
                f"Last phase '{self.phases[-1].name}' must end at 1.0, "
                f"got {last.value}"
            )


# ── Interaction ──────────────────────────────────────────────────────


class InjectionSpec(BaseModel):
    """Controls when the scaffolding prompt is re-injected."""

    frequency: int = Field(gt=0, description="Full re-injection every N turns.")
    reminder_frequency: int = Field(
        gt=0, description="Light reminder every N turns (excluding full injection turns)."
    )
    reminder_template: str


class AntiCapitulationRedirect(BaseModel):
    """A single trigger → replacement redirect rule."""

    trigger: str
    replacement: str


class AntiCapitulationSpec(BaseModel):
    """Rules to prevent the persona from agreeing with the interlocutor."""

    resistance_level: ResistanceLevel
    redirects: list[AntiCapitulationRedirect] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)


class ResponseLengthSpec(BaseModel):
    """Guidance on how long persona responses should be."""

    default: str
    by_phase: dict[str, str] = Field(default_factory=dict)


class StagnationDetectionSpec(BaseModel):
    """Runtime monitor config for agreement-loop detection."""

    enabled: bool = True
    window: int = Field(default=6, gt=0)
    similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    convergence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_turn: int = Field(default=10, ge=0)
    intervention_template: str = ""


class RepetitionDetectionSpec(BaseModel):
    """Runtime monitor config for formulaic output detection."""

    enabled: bool = True
    banned_patterns: list[str] = Field(default_factory=list)
    structural_patterns: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=0)


class InteractionSpec(BaseModel):
    """Full interaction protocol configuration."""

    injection: InjectionSpec
    anti_capitulation: AntiCapitulationSpec
    response_length: ResponseLengthSpec
    judge_window: int = Field(default=6, gt=0)
    stagnation_detection: StagnationDetectionSpec = Field(
        default_factory=StagnationDetectionSpec
    )
    repetition_detection: RepetitionDetectionSpec = Field(
        default_factory=RepetitionDetectionSpec
    )


# ── Evaluation ───────────────────────────────────────────────────────


class ScoringDimensionSpec(BaseModel):
    """Config for a single scoring dimension."""

    enabled: bool = True
    description: str = ""
    sample_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of turns to score (cost saving). None = all turns.",
    )


class ScoringSpec(BaseModel):
    """Which scoring dimensions are active and how they're configured."""

    persona_adherence: ScoringDimensionSpec = Field(
        default_factory=ScoringDimensionSpec
    )
    trajectory_adherence: ScoringDimensionSpec = Field(
        default_factory=ScoringDimensionSpec
    )
    naturalness: ScoringDimensionSpec = Field(
        default_factory=ScoringDimensionSpec
    )
    stagnation: ScoringDimensionSpec = Field(
        default_factory=ScoringDimensionSpec
    )
    fidelity: ScoringDimensionSpec = Field(
        default_factory=ScoringDimensionSpec
    )


class AdversarySpec(BaseModel):
    """Adversary configuration for evaluation runs."""

    type: AdversaryType = AdversaryType.NONE
    model: str | None = None


class EvaluationSpec(BaseModel):
    """Evaluation and scoring configuration."""

    scoring: ScoringSpec = Field(default_factory=ScoringSpec)
    adversary: AdversarySpec = Field(default_factory=AdversarySpec)


# ── Safety ───────────────────────────────────────────────────────────


class SafetySpec(BaseModel):
    """Safety boundaries for persona simulation."""

    intensity_ceiling: float = Field(default=0.9, gt=0.0, le=1.0)
    forbidden_simulation_content: list[str] = Field(default_factory=list)
    escalation_policy: str = ""
    persona_safety_note: str = ""


# ── Top-Level Config ─────────────────────────────────────────────────


class LoomConfig(BaseModel):
    """Root config model — validated from a YAML file."""

    schema_version: str
    persona: PersonaSpec
    trajectory: TrajectorySpec
    interaction: InteractionSpec
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)

    @model_validator(mode="after")
    def check_config(self) -> LoomConfig:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version '{self.schema_version}'. "
                f"Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        self._check_intensity_ceiling()
        self._check_phase_response_lengths()
        return self

    def _check_intensity_ceiling(self) -> None:
        """All dimension max_values must not exceed the safety ceiling."""
        ceiling = self.safety.intensity_ceiling
        for name, dim in self.trajectory.dimensions.items():
            if dim.max_value > ceiling:
                raise ValueError(
                    f"Dimension '{name}' max_value ({dim.max_value}) exceeds "
                    f"safety intensity_ceiling ({ceiling})"
                )

    def _check_phase_response_lengths(self) -> None:
        """If response_length.by_phase references a phase, it must exist."""
        phase_names = {p.name for p in self.trajectory.phases}
        for phase_name in self.interaction.response_length.by_phase:
            if phase_name not in phase_names:
                raise ValueError(
                    f"response_length.by_phase references unknown phase "
                    f"'{phase_name}'. Known phases: {sorted(phase_names)}"
                )
