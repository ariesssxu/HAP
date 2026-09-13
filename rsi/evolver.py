"""Reference harness mutation policy."""

from __future__ import annotations

from dataclasses import dataclass

from .harness import Harness
from .results import Diagnosis


@dataclass
class HarnessEvolver:
    """Applies small, auditable changes proposed by a diagnoser."""

    max_memory_items: int = 20
    max_workflow_actions: int = 20

    def evolve(self, harness: Harness, diagnosis: Diagnosis) -> Harness:
        if diagnosis.category in {"success", "unknown", "environment"}:
            return harness.clone()

        skills = list(dict.fromkeys(harness.skills + diagnosis.suggested_skills))
        memory = list(dict.fromkeys(harness.memory + diagnosis.memory_items))[-self.max_memory_items :]
        workflow = list(harness.workflow)
        for action in diagnosis.workflow_actions:
            if action not in workflow:
                insert_at = max(0, len(workflow) - 1) if "finish" in workflow else len(workflow)
                workflow.insert(insert_at, action)
        for skill in diagnosis.suggested_skills:
            if skill not in workflow and skill in {"inspect", "plan", "execute", "verify", "recover"}:
                insert_at = max(0, len(workflow) - 1) if "finish" in workflow else len(workflow)
                workflow.insert(insert_at, skill)
        workflow = workflow[: self.max_workflow_actions]
        note = f"Revision {harness.revision + 1}: respond to {diagnosis.category} failures."
        prompt = harness.prompt if note in harness.prompt else harness.prompt + " " + note
        return Harness(
            prompt=prompt,
            memory=memory,
            skills=skills,
            workflow=workflow,
            revision=harness.revision + 1,
            metadata={**harness.metadata, "last_diagnosis": diagnosis.category},
        )
