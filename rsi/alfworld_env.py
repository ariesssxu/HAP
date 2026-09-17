"""Optional ALFWorld text benchmark adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .harness import Harness
from .policy import StudentPolicy, WorkflowPolicy
from .results import EpisodeResult
from .specs import EnvironmentSpec, SpecValidationError, validate_spec


TASK_TYPES: Dict[str, int] = {
    "pick_and_place": 1,
    "look_in_light": 2,
    "clean_then_place": 3,
    "heat_then_place": 4,
    "cool_then_place": 5,
    "pick_two_then_place": 6,
}
SPLITS = {"train", "eval_in_distribution", "eval_out_of_distribution"}


def alfworld_spec(
    task_type: str,
    difficulty: int = 5,
    seed: int = 0,
    config_path: str = "configs/base_config.yaml",
    split: str = "train",
) -> EnvironmentSpec:
    if task_type not in TASK_TYPES:
        raise SpecValidationError(f"unknown ALFWorld task type {task_type!r}; choose {sorted(TASK_TYPES)}")
    if split not in SPLITS:
        raise SpecValidationError(f"unknown ALFWorld split {split!r}; choose {sorted(SPLITS)}")
    level = max(1, min(10, int(difficulty)))
    return EnvironmentSpec(
        name=f"alfworld-{task_type}-d{level}",
        domain="alfworld",
        goal=task_type.replace("_", " "),
        difficulty=level,
        horizon=20 + 10 * level,
        seed=seed,
        partial_observability=1.0,
        required_skills=["planning", "navigation", "manipulation"],
        action_space=["text_command"],
        metadata={
            "task_type": task_type,
            "task_type_id": TASK_TYPES[task_type],
            "split": split,
            "config_path": config_path,
        },
    )


@dataclass
class ALFWorldEnvironment:
    spec: EnvironmentSpec
    policy: StudentPolicy

    def run(self, harness: Harness, seed: int = 0) -> EpisodeResult:
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError("ALFWorld support requires: pip install -e '.[alfworld]' && alfworld-download") from exc

        config_path = str(self.spec.metadata["config_path"])
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["env"]["type"] = "AlfredTWEnv"
        config["env"]["task_types"] = [int(self.spec.metadata["task_type_id"])]
        split = str(self.spec.metadata["split"])
        wrapper = get_environment("AlfredTWEnv")(config, train_eval=split)
        env = wrapper.init_env(batch_size=1)
        if callable(getattr(env, "seed", None)):
            env.seed(self.spec.seed + seed)
        observations, infos = env.reset()
        observation = observations[0]
        self.policy.reset()
        trajectory: List[str] = []
        final_score = 0.0
        done = False
        for step in range(self.spec.horizon):
            actions = list(infos.get("admissible_commands", [[]])[0])
            if not actions:
                break
            action = self.policy.act(observation, harness, actions, step)
            if action not in actions:
                raise ValueError(f"policy returned inadmissible ALFWorld action: {action!r}")
            trajectory.append(action)
            observations, scores, dones, infos = env.step([action])
            observation, final_score, done = observations[0], float(scores[0]), bool(dones[0])
            if done:
                break
        if callable(getattr(env, "close", None)):
            env.close()
        won = infos.get("won", [final_score >= 1.0])
        success = bool(done and won[0])
        return EpisodeResult(
            success=success,
            score=max(0.0, min(1.0, final_score)),
            reward=final_score,
            steps=len(trajectory),
            trajectory=trajectory,
            observations=[str(observations[0]), "outcome: success" if success else "outcome: incomplete"],
            failure_reason=None if success else "policy_failed",
            oracle_signals={},
        )


class ALFWorldCompiler:
    domain = "alfworld"

    def __init__(self, policy: StudentPolicy | None = None, allowed_config_path: str | None = None) -> None:
        self.policy = policy or WorkflowPolicy()
        self.allowed_config_path = allowed_config_path

    def compile(self, spec: EnvironmentSpec) -> ALFWorldEnvironment:
        validate_spec(spec)
        if spec.domain != self.domain:
            raise SpecValidationError(f"ALFWorldCompiler cannot compile domain {spec.domain!r}")
        task_type = str(spec.metadata.get("task_type", ""))
        split = str(spec.metadata.get("split", ""))
        if task_type not in TASK_TYPES or spec.metadata.get("task_type_id") != TASK_TYPES[task_type]:
            raise SpecValidationError("ALFWorld task type and id must match the allow-list")
        if split not in SPLITS:
            raise SpecValidationError(f"unknown ALFWorld split {split!r}")
        if not isinstance(spec.metadata.get("config_path"), str):
            raise SpecValidationError("ALFWorld config_path must be a string")
        if self.allowed_config_path is None or spec.metadata["config_path"] != self.allowed_config_path:
            raise SpecValidationError("ALFWorld config_path is not the compiler-authorized path")
        return ALFWorldEnvironment(spec, self.policy)
