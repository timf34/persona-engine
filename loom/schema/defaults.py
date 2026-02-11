"""Enums, constants, and default values for Loom config schema."""

from enum import StrEnum

# Schema version — bump on breaking changes to config format.
SCHEMA_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSIONS = {"0.1.0"}

# Default safety ceiling for intensity dimensions.
DEFAULT_INTENSITY_CEILING = 0.9
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 0.9


class TrajectoryMode(StrEnum):
    """How the conversation length is determined."""

    FIXED_LENGTH = "fixed_length"
    # phase_gated and open_ended deferred to post-v0.


class CurveType(StrEnum):
    """Interpolation curve types for intensity dimensions."""

    SIGMOID = "sigmoid"
    LINEAR = "linear"
    STEP = "step"
    DELAYED_RAMP = "delayed_ramp"


class AdversaryType(StrEnum):
    """Types of adversarial interlocutors."""

    NONE = "none"
    CONTRADICTION = "contradiction"
    HELPFULNESS = "helpfulness"
    BOTH = "both"


class ResistanceLevel(StrEnum):
    """Anti-capitulation resistance strength."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EndConditionType(StrEnum):
    """How phase boundaries are determined."""

    PCT = "pct"
    TURN = "turn"
