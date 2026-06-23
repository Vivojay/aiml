from __future__ import annotations

from dataclasses import dataclass

from awr.types import Prediction, RewardSignal, Task, VerificationResult
from awr.utils import clamp


@dataclass(slots=True)
class RewardWeights:
    correctness: float = 0.5
    novelty: float = 0.12
    efficiency: float = 0.12
    consistency: float = 0.14
    difficulty: float = 0.12


class RewardEngine:
    """Converts verifier output and reasoning behavior into a scalar reward."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        task: Task,
        prediction: Prediction,
        verification: VerificationResult,
    ) -> RewardSignal:
        correctness = 1.0 if verification.correct else 0.0
        novelty = clamp(task.difficulty)
        step_count = max(1, len(prediction.trace.steps))
        efficiency = clamp(1.0 - max(0, step_count - 4) / 12.0)
        consistency = verification.confidence if verification.correct else max(0.0, verification.confidence - 0.4)
        difficulty = clamp(task.difficulty)
        total = (
            correctness * self.weights.correctness
            + novelty * self.weights.novelty
            + efficiency * self.weights.efficiency
            + consistency * self.weights.consistency
            + difficulty * self.weights.difficulty
        )
        return RewardSignal(
            correctness=correctness,
            novelty=novelty,
            efficiency=efficiency,
            consistency=clamp(consistency),
            difficulty=difficulty,
            total=clamp(total),
        )
