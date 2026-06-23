from __future__ import annotations

import random
from dataclasses import dataclass

from awr.types import Task, TaskKind, VerificationResult


@dataclass(slots=True)
class TaskStats:
    attempts: int = 0
    successes: int = 0
    average_confidence: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def record(self, verification: VerificationResult) -> None:
        self.attempts += 1
        if verification.correct:
            self.successes += 1
        self.average_confidence = (
            (self.average_confidence * (self.attempts - 1)) + verification.confidence
        ) / self.attempts


class SelfGeneratedTaskSystem:
    """Curriculum generator that focuses on weak task families."""

    task_kinds: tuple[TaskKind, ...] = ("arithmetic", "sequence", "logic")

    def __init__(self, *, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.counter = 0
        self.stats: dict[TaskKind, TaskStats] = {
            kind: TaskStats() for kind in self.task_kinds
        }

    def _next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter:05d}"

    def select_kind(self) -> TaskKind:
        weights = []
        for kind in self.task_kinds:
            stats = self.stats[kind]
            weakness = 1.0 - stats.accuracy if stats.attempts else 1.0
            uncertainty = 1.0 - stats.average_confidence if stats.attempts else 1.0
            weights.append(max(0.05, weakness + 0.5 * uncertainty))
        return self.random.choices(self.task_kinds, weights=weights, k=1)[0]

    def current_difficulty(self, kind: TaskKind) -> float:
        stats = self.stats[kind]
        if stats.attempts < 4:
            return 0.2
        return min(1.0, 0.2 + stats.accuracy * 0.65 + stats.attempts / 100.0)

    def generate(self, kind: TaskKind | None = None) -> Task:
        selected = kind or self.select_kind()
        difficulty = self.current_difficulty(selected)
        if selected == "arithmetic":
            return self._generate_arithmetic(difficulty)
        if selected == "sequence":
            return self._generate_sequence(difficulty)
        if selected == "logic":
            return self._generate_logic(difficulty)
        raise ValueError(f"unsupported generated task kind {selected}")

    def _number_range(self, difficulty: float) -> int:
        if difficulty < 0.35:
            return 12
        if difficulty < 0.7:
            return 50
        return 200

    def _generate_arithmetic(self, difficulty: float) -> Task:
        limit = self._number_range(difficulty)
        left = self.random.randint(1, limit)
        right = self.random.randint(1, limit)
        operators = ["+", "-"] if difficulty < 0.45 else ["+", "-", "*"]
        operator = self.random.choice(operators)
        if operator == "+":
            answer = left + right
        elif operator == "-":
            answer = left - right
        else:
            answer = left * right
        prompt = f"What is {left} {operator} {right}?"
        return Task(
            id=self._next_id("arith"),
            kind="arithmetic",
            prompt=prompt,
            expected_answer=str(answer),
            difficulty=difficulty,
            metadata={"left": left, "right": right, "operator": operator},
        )

    def _generate_sequence(self, difficulty: float) -> Task:
        start = self.random.randint(0, self._number_range(difficulty))
        delta = self.random.randint(1, 3 if difficulty < 0.45 else 12)
        length = 4 if difficulty < 0.7 else 5
        values = [start + delta * index for index in range(length)]
        answer = values[-1] + delta
        prompt = f"Complete the sequence: {', '.join(str(value) for value in values)}, ?"
        return Task(
            id=self._next_id("seq"),
            kind="sequence",
            prompt=prompt,
            expected_answer=str(answer),
            difficulty=difficulty,
            metadata={"start": start, "delta": delta, "length": length},
        )

    def _generate_logic(self, difficulty: float) -> Task:
        classes = [
            ("dax", "mip"),
            ("nars", "vel"),
            ("lumes", "tav"),
            ("brins", "sov"),
        ]
        people = ["nira", "kavi", "toma", "sena"]
        source, target = self.random.choice(classes)
        person = self.random.choice(people)
        ask_true = self.random.random() > (0.25 if difficulty < 0.7 else 0.45)
        asked_class = target if ask_true else self.random.choice(
            [item[1] for item in classes if item[1] != target]
        )
        answer = "yes" if ask_true else "no"
        prompt = f"All {source} are {target}. {person} is a {source}. Is {person} {asked_class}?"
        return Task(
            id=self._next_id("logic"),
            kind="logic",
            prompt=prompt,
            expected_answer=answer,
            difficulty=difficulty,
            metadata={"source": source, "target": target, "person": person},
        )

    def record_result(self, task: Task, verification: VerificationResult) -> None:
        if task.kind in self.stats:
            self.stats[task.kind].record(verification)
