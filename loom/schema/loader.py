"""Load and validate YAML configs into LoomConfig models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from loom.schema.config import LoomConfig


class ConfigError(Exception):
    """Raised when a config file cannot be loaded or validated."""


def load_config(path: str | Path) -> LoomConfig:
    """Read a YAML config file and return a validated LoomConfig.

    Raises ConfigError with a human-readable message on failure.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.suffix in (".yaml", ".yml"):
        raise ConfigError(f"Expected .yaml or .yml file, got: {path.suffix}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping at top level, got {type(data).__name__}")

    try:
        return LoomConfig.model_validate(data)
    except ValidationError as exc:
        # Re-format Pydantic errors into a readable string.
        lines = [f"Validation errors in {path.name}:"]
        for err in exc.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc
