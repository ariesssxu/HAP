"""Command line entry point: ``python -m rsi.cli``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .designers import (
    DifficultyDesigner,
    LearningProgressDesigner,
    LLMDesigner,
    MiniGridCurriculumDesigner,
    RandomDesigner,
    RandomMiniGridDesigner,
    spec_for_difficulty,
)
from .diagnosis import LLMDiagnoser, OracleDiagnoser, RuleBasedDiagnoser
from .evolver import HarnessEvolver
from .experiment import BASELINES, CoEvolutionExperiment, RunConfig
from .harness import Harness
from .llm import load_backend
from .minigrid_env import MiniGridCompiler, minigrid_spec
from .policy import LLMPolicy, load_policy
from .specs import CompilerRegistry, EnvironmentSpec
from .toy_env import ToyCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run environment/harness co-evolution baselines")
    parser.add_argument("--config", help="JSON RunConfig file")
    parser.add_argument("--rounds", type=int, help="override number of co-evolution rounds")
    parser.add_argument("--episodes-per-round", type=int, help="override rollouts per round")
    parser.add_argument("--seed", type=int, help="override random seed")
    parser.add_argument("--output-dir", help="override metrics directory")
    parser.add_argument("--domain", choices=("toy", "minigrid"), default="toy")
    parser.add_argument("--spec", help="fixed EnvironmentSpec JSON (default: built-in level 2)")
    parser.add_argument(
        "--designer",
        choices=("random", "difficulty", "learning_progress", "llm"),
        default="learning_progress",
        help="designer used by evolving-environment conditions",
    )
    parser.add_argument("--diagnoser", choices=("oracle", "rule_based", "llm"), default="oracle")
    parser.add_argument("--llm-backend", help="provider-neutral backend import path, module:object")
    parser.add_argument("--policy-backend", help="student policy import path, module:object")
    parser.add_argument("--llm-policy", action="store_true", help="use the configured text backend as a step policy")
    parser.add_argument(
        "--baseline",
        choices=("all",) + tuple(b.name for b in BASELINES),
        default="all",
    )
    parser.add_argument("--list-baselines", action="store_true")
    return parser


def _load_config(args: argparse.Namespace) -> RunConfig:
    value = {}
    if args.config:
        value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    overrides = {
        "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    value.update({key: item for key, item in overrides.items() if item is not None})
    return RunConfig.from_dict(value)


def _build_designer(name: str, seed: int, backend: object, fallback: EnvironmentSpec):
    if fallback.domain == "minigrid":
        if name == "llm":
            return LLMDesigner(backend, fallback=fallback)
        if name == "random":
            return RandomMiniGridDesigner(seed)
        return MiniGridCurriculumDesigner()
    if name == "random":
        return RandomDesigner(seed=seed)
    if name == "difficulty":
        return DifficultyDesigner(start=fallback.difficulty)
    if name == "learning_progress":
        return LearningProgressDesigner()
    return LLMDesigner(backend, fallback=fallback)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_baselines:
        for baseline in BASELINES:
            print(baseline.name)
        return 0

    config = _load_config(args)
    if args.spec:
        fixed_spec = EnvironmentSpec.from_json(args.spec)
    elif args.domain == "minigrid":
        fixed_spec = minigrid_spec("door_key", 5, config.seed, "fixed-minigrid")
    else:
        fixed_spec = spec_for_difficulty(4, config.seed, "fixed-toy")
    backend = load_backend(args.llm_backend)
    policy = LLMPolicy(backend) if args.llm_policy else load_policy(args.policy_backend)
    designer = _build_designer(args.designer, config.seed, backend, fixed_spec)
    diagnosers = {
        "oracle": OracleDiagnoser(),
        "rule_based": RuleBasedDiagnoser(),
        "llm": LLMDiagnoser(backend),
    }
    registry = CompilerRegistry([ToyCompiler(), MiniGridCompiler(policy)])
    experiment = CoEvolutionExperiment(
        registry=registry,
        fixed_spec=fixed_spec,
        evolving_designer=designer,
        diagnoser=diagnosers[args.diagnoser],
        evolver=HarnessEvolver(),
        initial_harness=Harness(),
        config=config,
    )
    selected = BASELINES if args.baseline == "all" else tuple(b for b in BASELINES if b.name == args.baseline)
    summaries = experiment.run_suite(selected)
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
