"""Main co-evolution loop and the four controlled baselines."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .designers import DesignContext, EnvironmentDesigner, FixedDesigner
from .diagnosis import Diagnoser
from .evolver import HarnessEvolver
from .harness import Harness
from .specs import CompilerRegistry, EnvironmentSpec


@dataclass(frozen=True)
class Baseline:
    name: str
    evolve_environment: bool
    evolve_harness: bool


BASELINES = (
    Baseline("fixed_env__fixed_harness", False, False),
    Baseline("evolving_env__fixed_harness", True, False),
    Baseline("fixed_env__evolving_harness", False, True),
    Baseline("evolving_env__evolving_harness", True, True),
)


@dataclass
class RunConfig:
    rounds: int = 12
    episodes_per_round: int = 3
    seed: int = 7
    output_dir: str = "results/rsi"

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RunConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown run config fields: {unknown}")
        config = cls(**value)
        if config.rounds < 1 or config.episodes_per_round < 1:
            raise ValueError("rounds and episodes_per_round must be positive")
        return config


@dataclass
class ExperimentSummary:
    baseline: str
    rounds: int
    episodes: int
    successes: int
    mean_score: float
    final_harness_revision: int
    final_difficulty: int
    log_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CoEvolutionExperiment:
    def __init__(
        self,
        registry: CompilerRegistry,
        fixed_spec: EnvironmentSpec,
        evolving_designer: EnvironmentDesigner,
        diagnoser: Diagnoser,
        evolver: HarnessEvolver,
        initial_harness: Harness,
        config: RunConfig,
    ) -> None:
        self.registry = registry
        self.fixed_spec = fixed_spec
        self.evolving_designer = evolving_designer
        self.diagnoser = diagnoser
        self.evolver = evolver
        self.initial_harness = initial_harness
        self.config = config

    def run(self, baseline: Baseline) -> ExperimentSummary:
        """Run one condition with isolated state and write one JSON object per episode."""

        rng = random.Random(self.config.seed)
        harness = self.initial_harness.clone()
        fixed_designer = FixedDesigner(self.fixed_spec)
        evolving_designer = copy.deepcopy(self.evolving_designer)
        scores: List[float] = []
        scores_by_difficulty: Dict[int, List[float]] = {}
        records: List[Dict[str, Any]] = []
        successes = 0
        last_spec = self.fixed_spec

        for round_index in range(self.config.rounds):
            designer = evolving_designer if baseline.evolve_environment else fixed_designer
            context = DesignContext(
                round_index=round_index,
                recent_scores=scores[-10:],
                scores_by_difficulty={key: values[-10:] for key, values in scores_by_difficulty.items()},
                last_spec=last_spec,
            )
            spec = designer.propose(context)
            environment = self.registry.compile(spec)
            last_spec = spec

            for episode_index in range(self.config.episodes_per_round):
                result = environment.run(harness, seed=rng.randrange(1_000_000))
                diagnosis = self.diagnoser.diagnose(spec, harness, result)
                successes += int(result.success)
                scores.append(result.score)
                scores_by_difficulty.setdefault(spec.difficulty, []).append(result.score)
                records.append(
                    {
                        "baseline": baseline.name,
                        "round": round_index,
                        "episode": episode_index,
                        "environment": spec.to_dict(),
                        "harness_revision": harness.revision,
                        "result": result.to_dict(),
                        "diagnosis": diagnosis.to_dict(),
                    }
                )
                if baseline.evolve_harness and not result.success:
                    harness = self.evolver.evolve(harness, diagnosis)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{baseline.name}.jsonl"
        log_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return ExperimentSummary(
            baseline=baseline.name,
            rounds=self.config.rounds,
            episodes=len(records),
            successes=successes,
            mean_score=sum(scores) / len(scores),
            final_harness_revision=harness.revision,
            final_difficulty=last_spec.difficulty,
            log_path=str(log_path),
        )

    def run_suite(self, baselines: Iterable[Baseline] = BASELINES) -> List[ExperimentSummary]:
        """Run all conditions. The caller should supply fresh stateful designers per suite."""

        summaries = [self.run(baseline) for baseline in baselines]
        output_dir = Path(self.config.output_dir)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps([summary.to_dict() for summary in summaries], indent=2) + "\n",
            encoding="utf-8",
        )
        return summaries
