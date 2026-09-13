"""Optional adapter to HAP's vendored MiniGrid implementation.

This module is intentionally lazy: importing ``rsi`` never imports gymnasium,
numpy, pygame, or MiniGrid.  It demonstrates the integration seam but is not
used by the zero-dependency toy experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Type

from .results import EpisodeResult
from .specs import EnvironmentSpec, SpecValidationError, validate_spec


class MiniGridEnvironment:
    ACTIONS: Dict[str, int] = {
        "left": 0,
        "right": 1,
        "forward": 2,
        "pickup": 3,
        "drop": 4,
        "toggle": 5,
        "finish": 6,
    }

    def __init__(self, spec: EnvironmentSpec, env: Any) -> None:
        self.spec = spec
        self.env = env

    def run(self, harness: Any, seed: int = 0) -> EpisodeResult:
        observation, _ = self.env.reset(seed=self.spec.seed + seed)
        trajectory = []
        observations = [str(observation.get("mission", self.spec.goal))]
        reward = 0.0
        terminated = truncated = False
        for token in harness.workflow[: self.spec.horizon]:
            if token not in self.ACTIONS:
                continue
            trajectory.append(token)
            observation, reward, terminated, truncated, _ = self.env.step(self.ACTIONS[token])
            if terminated or truncated:
                break
        success = bool(terminated and reward > 0)
        return EpisodeResult(
            success=success,
            score=float(reward),
            reward=float(reward),
            steps=len(trajectory),
            trajectory=trajectory,
            observations=observations + ["outcome: success" if success else "outcome: failed before goal"],
            failure_reason=None if success else "policy_failed",
            oracle_signals={},
        )


class MiniGridCompiler:
    domain = "minigrid"

    def compile(self, spec: EnvironmentSpec) -> MiniGridEnvironment:
        validate_spec(spec)
        if spec.domain != self.domain:
            raise SpecValidationError(f"MiniGridCompiler cannot compile domain '{spec.domain}'")

        vendor_root = Path(__file__).resolve().parents[1] / "envs" / "minigrid"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        try:
            from minigrid.envs import CrossingEnv, EmptyEnv, FourRoomsEnv
        except ImportError as exc:
            raise RuntimeError(
                "MiniGrid adapter needs the optional vendored dependencies. "
                "Install envs/minigrid (see rsi/README.md)."
            ) from exc

        variant = str(spec.metadata.get("variant", "empty"))
        size = int(spec.metadata.get("size", max(5, min(16, 4 + spec.difficulty))))
        if not 5 <= size <= 31:
            raise SpecValidationError("MiniGrid size must be in [5, 31]")
        variants: Dict[str, Type[Any]] = {
            "empty": EmptyEnv,
            "crossing": CrossingEnv,
            "four_rooms": FourRoomsEnv,
        }
        if variant not in variants:
            raise SpecValidationError(f"unknown MiniGrid variant '{variant}'; choose {sorted(variants)}")
        if variant == "crossing":
            size = size if size % 2 == 1 else size + 1
            env = CrossingEnv(size=size, num_crossings=max(1, min(3, spec.difficulty)), max_steps=spec.horizon)
        elif variant == "four_rooms":
            env = FourRoomsEnv(max_steps=spec.horizon)
        else:
            env = EmptyEnv(size=size, max_steps=spec.horizon)
        return MiniGridEnvironment(spec, env)
