"""Environment proposal baselines, from fixed curricula to LLM synthesis."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Protocol

from .llm import TextBackend
from .specs import EnvironmentSpec


SKILL_LADDER = ["inspect", "plan", "tool_use", "recover", "verify"]
RULE_LADDER = [
    "verify_before_finish",
    "inspect_before_execute",
    "recover_after_error",
]


@dataclass
class DesignContext:
    round_index: int
    recent_scores: List[float] = field(default_factory=list)
    scores_by_difficulty: Dict[int, List[float]] = field(default_factory=dict)
    last_spec: EnvironmentSpec | None = None


class EnvironmentDesigner(Protocol):
    name: str

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        ...


class FixedDesigner:
    name = "fixed"

    def __init__(self, spec: EnvironmentSpec) -> None:
        self.spec = spec

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        return self.spec


def spec_for_difficulty(difficulty: int, seed: int = 0, name: str | None = None) -> EnvironmentSpec:
    """Create a valid toy task on a controlled difficulty ladder."""

    level = max(1, min(10, int(difficulty)))
    skill_count = min(len(SKILL_LADDER), 1 + (level - 1) // 2)
    rule_count = min(len(RULE_LADDER), max(0, (level - 2) // 3))
    required_skills = SKILL_LADDER[:skill_count]
    rules = RULE_LADDER[:rule_count]
    horizon = max(3 + skill_count + rule_count, 5 + level // 2)
    actions = list(dict.fromkeys(["inspect", "plan", "execute", "verify", "recover", "finish"] + required_skills))
    return EnvironmentSpec(
        name=name or f"toy-d{level}",
        goal=f"complete a level-{level} synthetic task",
        difficulty=level,
        horizon=horizon,
        seed=seed,
        distractors=min(5, level // 2),
        partial_observability=min(1.0, 0.1 * (level - 1)),
        required_skills=required_skills,
        hidden_rules=rules,
        action_space=actions,
        metadata={"generator": "difficulty_ladder"},
    )


class RandomDesigner:
    name = "random"

    def __init__(self, seed: int = 0, min_difficulty: int = 1, max_difficulty: int = 10) -> None:
        self.rng = random.Random(seed)
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        level = self.rng.randint(self.min_difficulty, self.max_difficulty)
        return spec_for_difficulty(level, self.rng.randrange(1_000_000), f"random-r{context.round_index}-d{level}")


class DifficultyDesigner:
    """Increase difficulty after success and decrease it after failure."""

    name = "difficulty"

    def __init__(self, start: int = 1, target_score: float = 0.75) -> None:
        self.level = start
        self.target_score = target_score

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        if context.recent_scores:
            self.level += 1 if context.recent_scores[-1] >= self.target_score else -1
        self.level = max(1, min(10, self.level))
        return spec_for_difficulty(self.level, context.round_index, f"adaptive-r{context.round_index}-d{self.level}")


class LearningProgressDesigner:
    """Select the level with the greatest recent absolute learning progress.

    Unvisited levels are explored first.  Absolute progress deliberately values
    both learning and forgetting; replacing this rule is an obvious student
    research extension.
    """

    name = "learning_progress"

    def __init__(self, levels: range = range(1, 11)) -> None:
        self.levels = list(levels)

    @staticmethod
    def _progress(scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        midpoint = max(1, len(scores) // 2)
        old = sum(scores[:midpoint]) / midpoint
        new_values = scores[midpoint:]
        new = sum(new_values) / max(1, len(new_values))
        return abs(new - old)

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        unvisited = [level for level in self.levels if not context.scores_by_difficulty.get(level)]
        if unvisited:
            level = unvisited[0]
            return spec_for_difficulty(level, context.round_index, f"progress-r{context.round_index}-d{level}")
        level = max(
            self.levels,
            key=lambda candidate: (self._progress(context.scores_by_difficulty.get(candidate, [])), -candidate),
        )
        return spec_for_difficulty(level, context.round_index, f"progress-r{context.round_index}-d{level}")


class LLMDesigner:
    """Provider-neutral stub: ask for JSON, then pass it through strict validation."""

    name = "llm"

    def __init__(self, backend: TextBackend, fallback: EnvironmentSpec | None = None) -> None:
        self.backend = backend
        self.fallback = fallback

    def propose(self, context: DesignContext) -> EnvironmentSpec:
        prompt = (
            "Design one executable environment. Return only a JSON object matching "
            "EnvironmentSpec with fields name, domain, goal, difficulty, horizon, seed, "
            "distractors, partial_observability, required_skills, hidden_rules, "
            "action_space, metadata. domain must be toy or minigrid. "
            f"Round={context.round_index}; recent_scores={context.recent_scores[-5:]}"
        )
        try:
            value = json.loads(self.backend.complete(prompt))
            if not isinstance(value, dict):
                raise ValueError("backend response is not a JSON object")
            return EnvironmentSpec.from_dict(value)
        except Exception:
            if self.fallback is None:
                raise
            return replace(self.fallback, name=f"llm-fallback-r{context.round_index}", seed=context.round_index)
