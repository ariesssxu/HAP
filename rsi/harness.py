"""The mutable scaffolding around a fixed student policy/model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List


@dataclass
class Harness:
    """Prompt, memory, skills, and workflow that may evolve without weight updates."""

    prompt: str = "Inspect the task, make a plan, execute it, verify it, then finish."
    memory: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=lambda: ["inspect"])
    workflow: List[str] = field(
        default_factory=lambda: ["inspect", "plan", "execute", "verify", "finish"]
    )
    revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self, **changes: Any) -> "Harness":
        """Return an independent copy so baselines cannot leak state."""

        data = asdict(self)
        data.update(changes)
        return Harness(**data)

    def with_revision(self, **changes: Any) -> "Harness":
        return replace(self, revision=self.revision + 1, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
