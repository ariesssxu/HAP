"""Shared immutable records passed between environments, diagnosers, and logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    score: float
    reward: float
    steps: int
    trajectory: List[str]
    observations: List[str]
    failure_reason: Optional[str] = None
    oracle_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Diagnosis:
    category: str
    explanation: str
    suggested_skills: List[str] = field(default_factory=list)
    memory_items: List[str] = field(default_factory=list)
    workflow_actions: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
