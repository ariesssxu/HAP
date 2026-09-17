"""The mutable scaffolding around a fixed student policy/model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List


@dataclass
class Skill:
    """A reusable procedure with provenance and online validation statistics."""

    name: str
    instruction: str
    trigger: str = ""
    provenance: str = "diagnosis"
    successes: int = 0
    failures: int = 0

    @property
    def reliability(self) -> float:
        return (self.successes + 1) / (self.successes + self.failures + 2)


@dataclass
class Harness:
    """Prompt, memory, skills, and workflow that may evolve without weight updates."""

    prompt: str = "Inspect the task, make a plan, execute it, verify it, then finish."
    memory: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=lambda: ["inspect"])
    skill_library: Dict[str, Skill] = field(default_factory=dict)
    workflow: List[str] = field(
        default_factory=lambda: ["inspect", "plan", "execute", "verify", "finish"]
    )
    revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self, **changes: Any) -> "Harness":
        """Return an independent copy so baselines cannot leak state."""

        return replace(copy.deepcopy(self), **changes)

    def retrieve_skills(self, task: str, limit: int = 3) -> List[Skill]:
        """Return relevant, reliable procedures without an embedding dependency."""

        words = set(task.lower().replace("_", " ").split())
        ranked = sorted(
            self.skill_library.values(),
            key=lambda skill: (
                bool(words & set((skill.name + " " + skill.trigger).lower().replace("_", " ").split())),
                skill.reliability,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def with_revision(self, **changes: Any) -> "Harness":
        return replace(self, revision=self.revision + 1, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
