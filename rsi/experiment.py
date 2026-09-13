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
    validation_episodes: int = 2
    min_improvement: float = 0.0
    complexity_penalty: float = 0.01

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RunConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown run config fields: {unknown}")
        config = cls(**value)
        if config.rounds < 1 or config.episodes_per_round < 1 or config.validation_episodes < 1:
            raise ValueError("round and episode counts must be positive")
        if config.min_improvement < 0 or config.complexity_penalty < 0:
            raise ValueError("improvement threshold and complexity penalty must be non-negative")
        return config


@dataclass
class ExperimentSummary:
    baseline: str
    rounds: int
    episodes: int
    successes: int
    mean_score: float
    success_rate: float
    normalized_aulc: float
    difficulty_coverage: int
    task_coverage: int
    accepted_mutations: int
    proposed_mutations: int
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

    @staticmethod
    def _complexity(harness: Harness) -> int:
        return len(harness.skills) + len(harness.memory) + len(harness.workflow)

    @staticmethod
    def _aulc(scores: List[float]) -> float:
        if len(scores) == 1:
            return scores[0]
        return sum((left + right) / 2 for left, right in zip(scores, scores[1:])) / (len(scores) - 1)

    def _validate_mutation(self, environment: Any, incumbent: Harness, candidate: Harness, seed: int) -> Dict[str, Any]:
        """Evaluate both harnesses on identical seeds and apply a complexity-aware gate."""

        seeds = [seed + offset + 1 for offset in range(self.config.validation_episodes)]
        incumbent_score = sum(environment.run(incumbent, item).score for item in seeds) / len(seeds)
        candidate_score = sum(environment.run(candidate, item).score for item in seeds) / len(seeds)
        complexity_delta = max(0, self._complexity(candidate) - self._complexity(incumbent))
        required_gain = self.config.min_improvement + self.config.complexity_penalty * complexity_delta
        accepted = candidate.revision > incumbent.revision and candidate_score - incumbent_score > required_gain
        return {
            "accepted": accepted,
            "incumbent_score": incumbent_score,
            "candidate_score": candidate_score,
            "required_gain": required_gain,
            "validation_seeds": seeds,
        }

    def run(self, baseline: Baseline) -> ExperimentSummary:
        """Run one condition with isolated state and write one JSON object per episode."""

        rng = random.Random(self.config.seed)
        harness = self.initial_harness.clone()
        fixed_designer = FixedDesigner(self.fixed_spec)
        evolving_designer = copy.deepcopy(self.evolving_designer)
        scores: List[float] = []
        scores_by_difficulty: Dict[int, List[float]] = {}
        scores_by_task: Dict[str, List[float]] = {}
        records: List[Dict[str, Any]] = []
        successes = 0
        proposed_mutations = accepted_mutations = 0
        last_spec = self.fixed_spec

        for round_index in range(self.config.rounds):
            designer = evolving_designer if baseline.evolve_environment else fixed_designer
            context = DesignContext(
                round_index=round_index,
                recent_scores=scores[-10:],
                scores_by_difficulty={key: values[-10:] for key, values in scores_by_difficulty.items()},
                scores_by_task={key: values[-10:] for key, values in scores_by_task.items()},
                last_spec=last_spec,
            )
            spec = designer.propose(context)
            environment = self.registry.compile(spec)
            last_spec = spec

            for episode_index in range(self.config.episodes_per_round):
                rollout_seed = rng.randrange(1_000_000)
                result = environment.run(harness, seed=rollout_seed)
                diagnosis = self.diagnoser.diagnose(spec, harness, result)
                successes += int(result.success)
                scores.append(result.score)
                scores_by_difficulty.setdefault(spec.difficulty, []).append(result.score)
                scenario = str(spec.metadata.get("scenario", spec.name))
                task_key = f"{spec.domain}:{scenario}:{spec.difficulty}"
                scores_by_task.setdefault(task_key, []).append(result.score)
                record = {
                        "baseline": baseline.name,
                        "round": round_index,
                        "episode": episode_index,
                        "environment": spec.to_dict(),
                        "harness_revision": harness.revision,
                        "result": result.to_dict(),
                        "diagnosis": diagnosis.to_dict(),
                        "mutation_candidate": None,
                        "mutation_validation": None,
                    }
                if baseline.evolve_harness and not result.success:
                    candidate = self.evolver.evolve(harness, diagnosis)
                    decision = self._validate_mutation(environment, harness, candidate, rollout_seed)
                    record["mutation_candidate"] = candidate.to_dict()
                    record["mutation_validation"] = decision
                    if candidate.revision > harness.revision:
                        proposed_mutations += 1
                    if decision["accepted"]:
                        harness = candidate
                        accepted_mutations += 1
                records.append(record)

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
            success_rate=successes / len(records),
            normalized_aulc=self._aulc(scores),
            difficulty_coverage=len(scores_by_difficulty),
            task_coverage=len(scores_by_task),
            accepted_mutations=accepted_mutations,
            proposed_mutations=proposed_mutations,
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
