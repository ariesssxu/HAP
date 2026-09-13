"""Controlled environment and harness co-evolution research framework."""

from .harness import Harness
from .specs import EnvironmentSpec, SpecValidationError

__all__ = ["EnvironmentSpec", "Harness", "SpecValidationError"]
__version__ = "0.3.0"
