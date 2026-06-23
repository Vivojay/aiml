from __future__ import annotations

from dataclasses import dataclass, field

from awr.memory import MemoryStore
from awr.types import EpisodeRecord
from awr.utils import normalize_answer


@dataclass(slots=True)
class Belief:
    key: str
    value: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    retired: bool = False


class ContinualLearner:
    """Stores solved episodes, tracks beliefs, and supports corrections."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.beliefs: dict[str, Belief] = {}

    def learn_from_episode(self, episode: EpisodeRecord) -> list[str]:
        task = episode.task
        prediction = episode.prediction
        verification = episode.verification
        lessons: list[str] = []

        episodic = self.memory.add(
            "episodic",
            f"Problem: {task.prompt}\nAnswer: {prediction.answer}\nCorrect: {verification.correct}",
            metadata={
                "task_id": task.id,
                "prompt": task.prompt,
                "kind": task.kind,
                "answer": prediction.answer,
                "expected": task.expected_answer,
                "correct": verification.correct,
                "reward": episode.reward.total,
            },
            strength=1.0 + episode.reward.total,
        )

        if verification.correct:
            long_term = self.memory.add(
                "long_term",
                f"{task.prompt} -> {prediction.answer}",
                metadata={
                    "task_id": task.id,
                    "prompt": task.prompt,
                    "kind": task.kind,
                    "answer": prediction.answer,
                    "correct": True,
                    "source_episode": episodic.id,
                },
                strength=1.2 + episode.reward.total,
            )
            lessons.append(f"stored verified {task.kind} strategy in {long_term.id}")
            self._update_belief(task.prompt, prediction.answer, verification.confidence, long_term.id)
        else:
            self.memory.add(
                "short_term",
                f"Mistake on {task.prompt}: predicted {prediction.answer}; errors={verification.error_trace}",
                metadata={
                    "task_id": task.id,
                    "prompt": task.prompt,
                    "kind": task.kind,
                    "answer": prediction.answer,
                    "correct": False,
                },
                strength=0.6,
            )
            lessons.append("stored mistake for near-term rehearsal")

        return lessons

    def _update_belief(
        self,
        prompt: str,
        answer: str,
        confidence: float,
        evidence_id: str,
    ) -> None:
        key = normalize_answer(prompt)
        existing = self.beliefs.get(key)
        if existing is None or existing.retired:
            self.beliefs[key] = Belief(
                key=key,
                value=answer,
                confidence=confidence,
                evidence_ids=[evidence_id],
            )
            return

        if normalize_answer(existing.value) == normalize_answer(answer):
            existing.confidence = min(1.0, (existing.confidence + confidence) / 2.0 + 0.05)
            existing.evidence_ids.append(evidence_id)
        elif confidence >= existing.confidence:
            existing.retired = True
            self.beliefs[f"{key}#revision-{len(self.beliefs)}"] = Belief(
                key=key,
                value=answer,
                confidence=confidence,
                evidence_ids=[evidence_id],
            )

    def unlearn(self, prompt: str) -> bool:
        key = normalize_answer(prompt)
        belief = self.beliefs.get(key)
        if belief is None:
            return False
        belief.retired = True
        return True

    def rehearsal_batch(self, limit: int = 5) -> list[str]:
        hits = self.memory.retrieve("verified solved problem", kind="episodic", top_k=limit)
        return [hit.entry.text for hit in hits]
