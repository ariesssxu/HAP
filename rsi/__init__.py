"""Controlled environment and harness co-evolution research framework."""

from .harness import Harness, Skill
from .specs import EnvironmentSpec, SpecValidationError

__all__ = ["EnvironmentSpec", "Harness", "Skill", "SpecValidationError"]
__version__ = "0.4.0"
