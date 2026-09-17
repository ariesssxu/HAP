"""Reference harness mutation policy."""

from __future__ import annotations

from dataclasses import dataclass

from .harness import Harness, Skill
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
        skill_library = dict(harness.skill_library)
        for name in diagnosis.suggested_skills:
            skill_library.setdefault(
                name,
                Skill(name=name, instruction=diagnosis.explanation, trigger=diagnosis.category),
            )
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
            skill_library=skill_library,
            workflow=workflow,
            revision=harness.revision + 1,
            metadata={**harness.metadata, "last_diagnosis": diagnosis.category},
        )
