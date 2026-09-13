"""Deterministic local environment used by tests and classroom experiments."""

from __future__ import annotations

import random
from typing import Any, List

from .results import EpisodeResult
from .specs import EnvironmentSpec, SpecValidationError, validate_spec


class ToyEnvironment:
    """A transparent capability test, not a claim of realistic agency.

    The harness attempts its workflow.  Success requires the named capabilities,
    enough workflow budget, and memories of hidden constraints.  The environment
    emits both observable traces and privileged oracle signals so diagnosis
    methods can be compared fairly.
    """

    def __init__(self, spec: EnvironmentSpec) -> None:
        self.spec = spec

    def run(self, harness: Any, seed: int = 0) -> EpisodeResult:
        rng = random.Random(self.spec.seed + seed)
        workflow = list(harness.workflow[: self.spec.horizon])
        distractors = [f"distractor_{i}" for i in range(self.spec.distractors)]
        available = list(self.spec.action_space) + distractors
        rng.shuffle(available)

        observations: List[str] = [f"goal: {self.spec.goal}"]
        if self.spec.partial_observability < 1.0:
            observations.append("available actions: " + ", ".join(available))
        else:
            observations.append("available actions are hidden until inspection")

        missing_skills = [skill for skill in self.spec.required_skills if skill not in harness.skills]
        forgotten_rules = [
            rule
            for rule in self.spec.hidden_rules
            if rule not in harness.memory and rule.lower() not in harness.prompt.lower()
        ]
        missing_actions = [action for action in ("inspect", "execute", "finish") if action not in workflow]
        budget_needed = 3 + len(self.spec.required_skills) + len(self.spec.hidden_rules)
        budget_shortfall = max(0, budget_needed - self.spec.horizon)

        penalties = len(missing_skills) + len(forgotten_rules) + len(missing_actions) + budget_shortfall
        components = 3 + len(self.spec.required_skills) + len(self.spec.hidden_rules)
        score = max(0.0, min(1.0, 1.0 - penalties / max(components, 1)))
        success = penalties == 0

        if missing_skills:
            reason = "missing_skill"
        elif forgotten_rules:
            reason = "forgotten_hidden_rule"
        elif missing_actions:
            reason = "incomplete_workflow"
        elif budget_shortfall:
            reason = "horizon_too_short"
        else:
            reason = None

        observations.append("outcome: success" if success else "outcome: failed before goal")
        return EpisodeResult(
            success=success,
            score=score,
            reward=1.0 if success else score,
            steps=min(len(workflow), self.spec.horizon),
            trajectory=workflow,
            observations=observations,
            failure_reason=reason,
            oracle_signals={
                "missing_skills": missing_skills,
                "forgotten_rules": forgotten_rules,
                "missing_actions": missing_actions,
                "budget_shortfall": budget_shortfall,
            },
        )


class ToyCompiler:
    domain = "toy"

    def compile(self, spec: EnvironmentSpec) -> ToyEnvironment:
        validate_spec(spec)
        if spec.domain != self.domain:
            raise SpecValidationError(f"ToyCompiler cannot compile domain '{spec.domain}'")
        return ToyEnvironment(spec)
