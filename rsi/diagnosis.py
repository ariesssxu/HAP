"""Failure attribution interfaces and baseline implementations."""

from __future__ import annotations

import json
from typing import Protocol

from .harness import Harness
from .llm import TextBackend
from .results import Diagnosis, EpisodeResult
from .specs import EnvironmentSpec


class Diagnoser(Protocol):
    name: str

    def diagnose(self, spec: EnvironmentSpec, harness: Harness, result: EpisodeResult) -> Diagnosis:
        ...


class OracleDiagnoser:
    """Uses privileged simulator signals; an upper-bound attribution baseline."""

    name = "oracle"

    def diagnose(self, spec: EnvironmentSpec, harness: Harness, result: EpisodeResult) -> Diagnosis:
        signals = result.oracle_signals
        if result.success:
            return Diagnosis("success", "No mutation is required.")
        if signals.get("missing_skills"):
            skills = list(signals["missing_skills"])
            return Diagnosis("capability", f"Missing required skills: {skills}", suggested_skills=skills)
        if signals.get("forgotten_rules"):
            rules = list(signals["forgotten_rules"])
            return Diagnosis("memory", f"Hidden constraints were forgotten: {rules}", memory_items=rules)
        if signals.get("missing_actions"):
            actions = list(signals["missing_actions"])
            return Diagnosis("workflow", f"Workflow omitted actions: {actions}", workflow_actions=actions)
        return Diagnosis("environment", "The task horizon is shorter than its required procedure.", confidence=1.0)


class RuleBasedDiagnoser:
    """Infers a repair from observable outcome and the public task spec."""

    name = "rule_based"

    def diagnose(self, spec: EnvironmentSpec, harness: Harness, result: EpisodeResult) -> Diagnosis:
        if result.success:
            return Diagnosis("success", "Observable rollout succeeded.")
        missing = [skill for skill in spec.required_skills if skill not in harness.skills]
        if missing:
            return Diagnosis("capability", "Public spec names skills absent from the harness.", suggested_skills=missing, confidence=0.8)
        if spec.hidden_rules and "outcome: failed before goal" in result.observations:
            return Diagnosis("memory", "Failure is consistent with an untracked constraint.", memory_items=spec.hidden_rules, confidence=0.6)
        missing_actions = [a for a in ("inspect", "execute", "finish") if a not in result.trajectory]
        if missing_actions:
            return Diagnosis("workflow", "Observed trace omitted a core action.", workflow_actions=missing_actions, confidence=0.8)
        return Diagnosis("unknown", "The observable trace does not identify a unique cause.", confidence=0.2)


class LLMDiagnoser:
    """Provider-neutral JSON-in/JSON-out diagnosis stub."""

    name = "llm"

    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def diagnose(self, spec: EnvironmentSpec, harness: Harness, result: EpisodeResult) -> Diagnosis:
        payload = {"spec": spec.to_dict(), "harness": harness.to_dict(), "result": result.to_dict()}
        response = self.backend.complete(
            "Diagnose this failed rollout. Return only JSON with category, explanation, "
            "suggested_skills, memory_items, workflow_actions, confidence.\n" + json.dumps(payload)
        )
        value = json.loads(response)
        if not isinstance(value, dict):
            raise ValueError("LLM diagnosis must be a JSON object")
        return Diagnosis(**value)
