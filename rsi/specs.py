"""Environment DSL, validation, and compiler registry.

JSON is used as the on-disk DSL because it is in Python's standard library and
is easy for both students and language models to emit.  The important research
boundary is: a designer proposes data, a validator rejects unsafe/invalid data,
and a deterministic compiler creates an executable environment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Protocol


class SpecValidationError(ValueError):
    """Raised when an environment proposal is not executable."""


@dataclass(frozen=True)
class EnvironmentSpec:
    """Small, serializable language for proposing environments."""

    name: str
    domain: str = "toy"
    goal: str = "complete the task"
    difficulty: int = 1
    horizon: int = 6
    seed: int = 0
    distractors: int = 0
    partial_observability: float = 0.0
    required_skills: List[str] = field(default_factory=lambda: ["inspect"])
    hidden_rules: List[str] = field(default_factory=list)
    action_space: List[str] = field(
        default_factory=lambda: ["inspect", "plan", "execute", "verify", "finish"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentSpec":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise SpecValidationError(f"unknown EnvironmentSpec fields: {unknown}")
        try:
            spec = cls(**dict(value))
        except TypeError as exc:
            raise SpecValidationError(str(exc)) from exc
        validate_spec(spec)
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "EnvironmentSpec":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecValidationError(f"could not read spec {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise SpecValidationError("the top-level JSON value must be an object")
        return cls.from_dict(value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _nonempty_strings(name: str, values: Any) -> List[str]:
    if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
        raise SpecValidationError(f"{name} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise SpecValidationError(f"{name} must not contain duplicates")
    return values


def validate_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    """Validate all invariants before any compiler sees a proposal."""

    if not isinstance(spec.name, str) or not spec.name.strip():
        raise SpecValidationError("name must be a non-empty string")
    if not isinstance(spec.domain, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", spec.domain):
        raise SpecValidationError("domain must be a lowercase identifier")
    if not isinstance(spec.goal, str) or not spec.goal.strip():
        raise SpecValidationError("goal must be a non-empty string")
    if not isinstance(spec.difficulty, int) or not 1 <= spec.difficulty <= 10:
        raise SpecValidationError("difficulty must be an integer in [1, 10]")
    if not isinstance(spec.horizon, int) or not 2 <= spec.horizon <= 1000:
        raise SpecValidationError("horizon must be an integer in [2, 1000]")
    if not isinstance(spec.seed, int):
        raise SpecValidationError("seed must be an integer")
    if not isinstance(spec.distractors, int) or not 0 <= spec.distractors <= 50:
        raise SpecValidationError("distractors must be an integer in [0, 50]")
    if not isinstance(spec.partial_observability, (int, float)) or not 0.0 <= spec.partial_observability <= 1.0:
        raise SpecValidationError("partial_observability must be in [0, 1]")
    _nonempty_strings("required_skills", spec.required_skills)
    _nonempty_strings("hidden_rules", spec.hidden_rules)
    actions = _nonempty_strings("action_space", spec.action_space)
    if "finish" not in actions and spec.domain == "toy":
        raise SpecValidationError("toy action_space must contain 'finish'")
    if not isinstance(spec.metadata, dict):
        raise SpecValidationError("metadata must be an object")
    return spec


class CompiledEnvironment(Protocol):
    """Runtime interface consumed by the co-evolution loop."""

    spec: EnvironmentSpec

    def run(self, harness: Any, seed: int = 0) -> Any:
        ...


class EnvironmentCompiler(Protocol):
    domain: str

    def compile(self, spec: EnvironmentSpec) -> CompiledEnvironment:
        ...


class CompilerRegistry:
    """Maps validated DSL domains to deterministic compilers."""

    def __init__(self, compilers: Iterable[EnvironmentCompiler] = ()) -> None:
        self._compilers = {compiler.domain: compiler for compiler in compilers}

    def register(self, compiler: EnvironmentCompiler) -> None:
        self._compilers[compiler.domain] = compiler

    def compile(self, spec: EnvironmentSpec) -> CompiledEnvironment:
        validate_spec(spec)
        try:
            compiler = self._compilers[spec.domain]
        except KeyError as exc:
            raise SpecValidationError(f"no compiler registered for domain '{spec.domain}'") from exc
        return compiler.compile(spec)
