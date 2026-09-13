"""MiniGrid scenario compiler with no import-time optional dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .harness import Harness
from .policy import StudentPolicy, WorkflowPolicy
from .results import EpisodeResult
from .specs import EnvironmentSpec, SpecValidationError, validate_spec


SCENARIOS: Dict[str, Sequence[str]] = {
    "empty": ("MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0"),
    "door_key": ("MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0"),
    "dynamic_obstacles": (
        "MiniGrid-Dynamic-Obstacles-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-16x16-v0",
    ),
    "lava_crossing": (
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS11N5-v0",
    ),
    "multi_room": ("MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v1", "MiniGrid-MultiRoom-N6-v0"),
    "four_rooms": ("MiniGrid-FourRooms-v0",) * 3,
    "memory": ("MiniGrid-MemoryS7-v0", "MiniGrid-MemoryS11-v0", "MiniGrid-MemoryS17Random-v0"),
    "fetch": ("MiniGrid-Fetch-5x5-N2-v0", "MiniGrid-Fetch-8x8-N3-v0", "MiniGrid-Fetch-8x8-N3-v0"),
    "put_near": ("MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-PutNear-8x8-N3-v0", "MiniGrid-PutNear-8x8-N3-v0"),
    "red_blue_doors": ("MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-8x8-v0", "MiniGrid-RedBlueDoors-8x8-v0"),
    "key_corridor": ("MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS6R3-v0"),
    "unlock_pickup": ("MiniGrid-UnlockPickup-v0",) * 3,
    "blocked_unlock_pickup": ("MiniGrid-BlockedUnlockPickup-v0",) * 3,
}

ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def minigrid_spec(scenario: str, difficulty: int, seed: int = 0, name: str | None = None) -> EnvironmentSpec:
    if scenario not in SCENARIOS:
        raise SpecValidationError(f"unknown MiniGrid scenario {scenario!r}; choose {sorted(SCENARIOS)}")
    level = max(1, min(10, int(difficulty)))
    tier = min(2, (level - 1) // 3)
    return EnvironmentSpec(
        name=name or f"minigrid-{scenario}-d{level}",
        domain="minigrid",
        goal=f"solve the {scenario.replace('_', ' ')} mission",
        difficulty=level,
        horizon=40 + 20 * level,
        seed=seed,
        distractors=level // 2,
        partial_observability=1.0,
        required_skills=["navigation"]
        + (["memory"] if scenario in {"memory", "red_blue_doors"} else [])
        + (
            ["manipulation"]
            if scenario in {"door_key", "fetch", "put_near", "key_corridor", "unlock_pickup", "blocked_unlock_pickup"}
            else []
        ),
        hidden_rules=[],
        action_space=list(ACTIONS),
        metadata={"scenario": scenario, "tier": tier, "env_id": SCENARIOS[scenario][tier]},
    )


@dataclass
class MiniGridEnvironment:
    spec: EnvironmentSpec
    env_id: str
    policy: StudentPolicy

    def run(self, harness: Harness, seed: int = 0) -> EpisodeResult:
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 - registers environments with Gymnasium
        except ImportError as exc:
            raise RuntimeError("MiniGrid support requires: pip install -e '.[minigrid]'") from exc

        env = gym.make(self.env_id, max_steps=self.spec.horizon)
        observation, _ = env.reset(seed=self.spec.seed + seed)
        self.policy.reset()
        trajectory: List[str] = []
        rewards = 0.0
        terminated = truncated = False
        mission = str(observation.get("mission", self.spec.goal))
        for step in range(self.spec.horizon):
            action = self.policy.act(observation, harness, ACTIONS, step)
            if action not in ACTIONS:
                raise ValueError(f"policy returned invalid MiniGrid action: {action!r}")
            trajectory.append(action)
            observation, reward, terminated, truncated, _ = env.step(ACTIONS.index(action))
            rewards += float(reward)
            if terminated or truncated:
                break
        env.close()
        success = bool(terminated and rewards > 0)
        return EpisodeResult(
            success=success,
            score=max(0.0, min(1.0, rewards)),
            reward=rewards,
            steps=len(trajectory),
            trajectory=trajectory,
            observations=[f"mission: {mission}", "outcome: success" if success else "outcome: timeout or failure"],
            failure_reason=None if success else "policy_failed",
            oracle_signals={},
        )


class MiniGridCompiler:
    domain = "minigrid"

    def __init__(self, policy: StudentPolicy | None = None) -> None:
        self.policy = policy or WorkflowPolicy()

    def compile(self, spec: EnvironmentSpec) -> MiniGridEnvironment:
        validate_spec(spec)
        if spec.domain != self.domain:
            raise SpecValidationError(f"MiniGridCompiler cannot compile domain {spec.domain!r}")
        scenario = str(spec.metadata.get("scenario", ""))
        if scenario not in SCENARIOS:
            raise SpecValidationError(f"unknown MiniGrid scenario {scenario!r}; choose {sorted(SCENARIOS)}")
        tier = spec.metadata.get("tier", min(2, (spec.difficulty - 1) // 3))
        if not isinstance(tier, int) or not 0 <= tier <= 2:
            raise SpecValidationError("MiniGrid tier must be an integer in [0, 2]")
        expected_id = SCENARIOS[scenario][tier]
        env_id = str(spec.metadata.get("env_id", expected_id))
        if env_id != expected_id:
            raise SpecValidationError("env_id must match the registered scenario and tier")
        return MiniGridEnvironment(spec, env_id, self.policy)
