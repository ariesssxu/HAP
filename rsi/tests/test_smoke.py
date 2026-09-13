from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rsi.designers import DesignContext, LearningProgressDesigner, spec_for_difficulty
from rsi.diagnosis import OracleDiagnoser
from rsi.evolver import HarnessEvolver
from rsi.experiment import BASELINES, CoEvolutionExperiment, RunConfig
from rsi.harness import Harness
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

    def test_learning_progress_explores_unvisited_levels(self) -> None:
        designer = LearningProgressDesigner(range(1, 4))
        context = DesignContext(0, scores_by_difficulty={1: [0.1, 0.8]})
        self.assertEqual(designer.propose(context).difficulty, 2)

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
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual({item["baseline"] for item in summary}, {item.name for item in BASELINES})


if __name__ == "__main__":
    unittest.main()
