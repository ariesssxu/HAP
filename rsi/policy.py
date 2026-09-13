"""Observation-conditioned student policies used inside compiled environments."""

from __future__ import annotations

import importlib
import json
from typing import Any, Protocol, Sequence

from .harness import Harness
from .llm import TextBackend


class StudentPolicy(Protocol):
    def reset(self) -> None:
        ...

    def act(
        self,
        observation: Any,
        harness: Harness,
        actions: Sequence[str],
        step: int,
    ) -> str:
        ...


class WorkflowPolicy:
    """Small deterministic control baseline; not intended as a strong solver."""

    def reset(self) -> None:
        pass

    def act(self, observation: Any, harness: Harness, actions: Sequence[str], step: int) -> str:
        workflow = [item for item in harness.workflow if item in actions]
        return workflow[step % len(workflow)] if workflow else "forward"


class LLMPolicy:
    """Ask a text backend for one action conditioned on the current observation."""

    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def reset(self) -> None:
        pass

    def act(self, observation: Any, harness: Harness, actions: Sequence[str], step: int) -> str:
        payload = {
            "observation": _jsonable(observation),
            "harness": harness.to_dict(),
            "actions": list(actions),
            "step": step,
        }
        response = self.backend.complete(
            "Choose one action. Return only its exact name.\n" + json.dumps(payload)
        ).strip()
        if response not in actions:
            raise ValueError(f"policy returned invalid action: {response!r}")
        return response


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def load_policy(import_path: str | None) -> StudentPolicy:
    """Load ``module:object`` or return the deterministic workflow baseline."""

    if not import_path:
        return WorkflowPolicy()
    if ":" not in import_path:
        raise ValueError("policy path must have the form module:object")
    module_name, object_name = import_path.split(":", 1)
    value = getattr(importlib.import_module(module_name), object_name)
    policy = value() if isinstance(value, type) else value
    if not callable(getattr(policy, "act", None)) or not callable(getattr(policy, "reset", None)):
        raise TypeError(f"{import_path} must expose reset() and act(observation, harness, actions, step)")
    return policy
