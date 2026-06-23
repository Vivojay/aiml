from __future__ import annotations

from awr.memory import MemoryStore
from awr.types import Prediction, Task, VerificationResult
from awr.utils import normalize_answer, numeric_value


def _answers_match(left: str | None, right: str | None) -> bool:
    left_numeric = numeric_value(left)
    right_numeric = numeric_value(right)
    if left_numeric is not None and right_numeric is not None:
        return abs(left_numeric - right_numeric) < 1e-9
    return normalize_answer(left) == normalize_answer(right)


class ExactVerifier:
    name = "exact"

    def verify(self, task: Task, prediction: Prediction) -> VerificationResult:
        if task.expected_answer is None:
            return VerificationResult(
                correct=False,
                confidence=0.0,
                verifier_name=self.name,
                error_trace=["no expected answer available for exact verification"],
            )

        correct = _answers_match(prediction.answer, task.expected_answer)
        return VerificationResult(
            correct=correct,
            confidence=0.98 if correct else 0.95,
            verifier_name=self.name,
            error_trace=[] if correct else [
                f"expected {task.expected_answer!r}, got {prediction.answer!r}"
            ],
            details={"expected": task.expected_answer, "actual": prediction.answer},
        )


class ConsistencyVerifier:
    name = "consistency"

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def verify(self, task: Task, prediction: Prediction) -> VerificationResult:
        hits = self.memory.retrieve(task.prompt, kind="long_term", top_k=3, min_score=0.82)
        conflicts: list[str] = []
        task_prompt = normalize_answer(task.prompt)
        for hit in hits:
            remembered_prompt = hit.entry.metadata.get("prompt")
            if remembered_prompt is None:
                continue
            if normalize_answer(str(remembered_prompt)) != task_prompt:
                continue
            remembered = hit.entry.metadata.get("answer")
            if remembered is None:
                continue
            if not _answers_match(str(remembered), prediction.answer):
                conflicts.append(
                    f"memory {hit.entry.id} says {remembered!r}; prediction says {prediction.answer!r}"
                )

        return VerificationResult(
            correct=not conflicts,
            confidence=0.8 if not conflicts else 0.35,
            verifier_name=self.name,
            error_trace=conflicts,
            details={"checked_memories": [hit.entry.id for hit in hits]},
        )


class CompositeVerifier:
    """Combines exact task feedback with memory consistency checks."""

    def __init__(self, memory: MemoryStore) -> None:
        self.exact = ExactVerifier()
        self.consistency = ConsistencyVerifier(memory)

    def verify(self, task: Task, prediction: Prediction) -> VerificationResult:
        consistency = self.consistency.verify(task, prediction)
        if task.expected_answer is None:
            return consistency

        exact = self.exact.verify(task, prediction)
        correct = exact.correct and consistency.correct
        confidence = min(exact.confidence, consistency.confidence if not consistency.correct else 1.0)
        errors = [*exact.error_trace, *consistency.error_trace]
        return VerificationResult(
            correct=correct,
            confidence=confidence,
            verifier_name="composite",
            error_trace=errors,
            details={
                "exact": exact.details,
                "consistency": consistency.details,
            },
        )
