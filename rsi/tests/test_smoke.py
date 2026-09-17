from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rsi.designers import DesignContext, LearningProgressDesigner, MiniGridCurriculumDesigner, spec_for_difficulty
from rsi.alfworld_env import ALFWorldCompiler, TASK_TYPES, alfworld_spec
from rsi.diagnosis import OracleDiagnoser
from rsi.evolver import HarnessEvolver
from rsi.experiment import BASELINES, Baseline, CoEvolutionExperiment, RunConfig
from rsi.harness import Harness, Skill
from rsi.minigrid_env import MiniGridCompiler, SCENARIOS, minigrid_spec
from rsi.policy import WorkflowPolicy
from rsi.specs import CompilerRegistry, EnvironmentSpec, SpecValidationError
from rsi.toy_env import ToyCompiler


class SpecTests(unittest.TestCase):
    def test_json_round_trip_and_compile(self) -> None:
        spec = spec_for_difficulty(3)
        rebuilt = EnvironmentSpec.from_dict(spec.to_dict())
        env = CompilerRegistry([ToyCompiler()]).compile(rebuilt)
        self.assertEqual(env.spec, spec)

    def test_validator_rejects_unknown_fields(self) -> None:
        value = spec_for_difficulty(1).to_dict()
        value["python_code"] = "import os"
        with self.assertRaises(SpecValidationError):
            EnvironmentSpec.from_dict(value)

    def test_minigrid_scenarios_compile_without_importing_dependency(self) -> None:
        compiler = MiniGridCompiler(WorkflowPolicy())
        for scenario in SCENARIOS:
            environment = compiler.compile(minigrid_spec(scenario, 5))
            self.assertEqual(environment.spec.metadata["scenario"], scenario)

    def test_alfworld_tasks_compile_without_importing_dependency(self) -> None:
        compiler = ALFWorldCompiler(WorkflowPolicy(), "base.yaml")
        for task_type in TASK_TYPES:
            environment = compiler.compile(alfworld_spec(task_type, config_path="base.yaml"))
            self.assertEqual(environment.spec.metadata["task_type"], task_type)


class HarnessTests(unittest.TestCase):
    def test_skill_retrieval_prefers_relevant_reliable_procedure(self) -> None:
        harness = Harness(
            skill_library={
                "door": Skill("door", "pick up key then toggle", "door key", successes=4),
                "lava": Skill("lava", "avoid red cells", "hazard", successes=10),
            }
        )
        self.assertEqual(harness.retrieve_skills("open the door with a key", limit=1)[0].name, "door")


class EvolutionTests(unittest.TestCase):
    def test_oracle_mutation_repairs_toy_harness(self) -> None:
        spec = spec_for_difficulty(4)
        env = ToyCompiler().compile(spec)
        harness = Harness()
        first = env.run(harness)
        diagnosis = OracleDiagnoser().diagnose(spec, harness, first)
        changed = HarnessEvolver().evolve(harness, diagnosis)
        self.assertGreater(changed.revision, harness.revision)
        self.assertTrue(set(diagnosis.suggested_skills).issubset(changed.skills))

    def test_learning_progress_targets_competence_frontier(self) -> None:
        designer = LearningProgressDesigner(range(1, 4))
        context = DesignContext(
            0,
            scores_by_difficulty={1: [1.0, 1.0], 2: [0.65, 0.7], 3: [0.0, 0.0]},
        )
        self.assertEqual(designer.propose(context).difficulty, 2)

    def test_minigrid_curriculum_uses_per_scenario_evidence(self) -> None:
        designer = MiniGridCurriculumDesigner(scenarios=["empty", "door_key"], levels=range(5, 6))
        context = DesignContext(
            1,
            scores_by_task={"minigrid:empty:5": [1.0] * 8},
        )
        self.assertEqual(designer.propose(context).metadata["scenario"], "door_key")

    def test_mutation_gate_rejects_complexity_without_gain(self) -> None:
        experiment = CoEvolutionExperiment(
            registry=CompilerRegistry([ToyCompiler()]),
            fixed_spec=spec_for_difficulty(1),
            evolving_designer=LearningProgressDesigner(range(1, 4)),
            diagnoser=OracleDiagnoser(),
            evolver=HarnessEvolver(),
            initial_harness=Harness(),
            config=RunConfig(validation_episodes=2, complexity_penalty=0.01),
        )
        environment = ToyCompiler().compile(spec_for_difficulty(1))
        incumbent = Harness()
        candidate = incumbent.clone(skills=incumbent.skills + ["unused"], revision=1)
        decision = experiment._validate_mutation(environment, incumbent, candidate, seed=13)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["candidate_score"], decision["incumbent_score"])

    def test_all_four_baselines_write_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = CoEvolutionExperiment(
                registry=CompilerRegistry([ToyCompiler()]),
                fixed_spec=spec_for_difficulty(3),
                evolving_designer=LearningProgressDesigner(range(1, 4)),
                diagnoser=OracleDiagnoser(),
                evolver=HarnessEvolver(),
                initial_harness=Harness(),
                config=RunConfig(rounds=2, episodes_per_round=1, output_dir=directory),
            )
            summaries = experiment.run_suite()
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(Path(summary.log_path).exists() for summary in summaries))
            self.assertTrue(all(0.0 <= summary.normalized_aulc <= 1.0 for summary in summaries))
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual({item["baseline"] for item in summary}, {item.name for item in BASELINES})

    def test_retention_control_isolates_cross_round_learning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = CoEvolutionExperiment(
                registry=CompilerRegistry([ToyCompiler()]),
                fixed_spec=spec_for_difficulty(4),
                evolving_designer=LearningProgressDesigner(range(1, 4)),
                diagnoser=OracleDiagnoser(),
                evolver=HarnessEvolver(),
                initial_harness=Harness(),
                config=RunConfig(rounds=2, episodes_per_round=1, output_dir=directory),
            )
            baseline = Baseline("retention_test", False, True)
            summaries = experiment.run_suite([baseline], ("retained", "reset"))
            by_mode = {summary.retention: summary for summary in summaries}
            self.assertGreater(by_mode["retained"].successes, by_mode["reset"].successes)


if __name__ == "__main__":
    unittest.main()
