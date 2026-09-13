"""Student-friendly environment and harness co-evolution prototype.

The original HAP implementation remains under ``src/`` and ``envs/``.  This
package is deliberately independent so that it can be read and run without the
large experimental dependency set used by the paper code.
"""

from .harness import Harness
from .specs import EnvironmentSpec, SpecValidationError

__all__ = ["EnvironmentSpec", "Harness", "SpecValidationError"]
__version__ = "0.1.0"
