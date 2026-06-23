from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MemoryKind = Literal["short_term", "long_term", "episodic"]
TaskKind = Literal["arithmetic", "sequence", "logic", "language", "unknown"]


@dataclass(slots=True)
class Observation:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LatentState:
    vector: tuple[float, ...]
    symbols: frozenset[str]
    source_text: str
    confidence: float


@dataclass(slots=True)
class Task:
    id: str
    kind: TaskKind
    prompt: str
    expected_answer: str | None = None
    difficulty: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    strategy: str
    target_experts: tuple[str, ...]
    internal_cycles: int
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class ReasoningTrace:
    plan: Plan
    steps: list[str] = field(default_factory=list)
    selected_memory_ids: list[str] = field(default_factory=list)
    selected_experts: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class Prediction:
    answer: str
    confidence: float
    latent_state: LatentState
    trace: ReasoningTrace
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    correct: bool
    confidence: float
    verifier_name: str
    error_trace: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardSignal:
    correctness: float
    novelty: float
    efficiency: float
    consistency: float
    difficulty: float
    total: float


@dataclass(slots=True)
class AdaptationEvent:
    action: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeRecord:
    task: Task
    prediction: Prediction
    verification: VerificationResult
    reward: RewardSignal
    adaptation_events: list[AdaptationEvent] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExpertResult:
    expert_name: str
    answer: str
    confidence: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)
