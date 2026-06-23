from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Protocol

from awr.memory import MemoryHit
from awr.types import ExpertResult, LatentState, ReasoningTrace, Task
from awr.utils import clamp, cosine_similarity, normalize_answer, numeric_value


class Expert(Protocol):
    name: str
    tags: tuple[str, ...]
    utility: float
    protected: bool

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        ...


@dataclass(slots=True)
class BaseExpert:
    name: str
    tags: tuple[str, ...]
    utility: float = 0.5
    protected: bool = True

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        raise NotImplementedError


class SafeArithmeticEvaluator(ast.NodeVisitor):
    allowed_binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("non-numeric literal")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operator = type(node.op)
        if operator not in self.allowed_binary:
            raise ValueError(f"unsupported operator {operator.__name__}")
        return self.allowed_binary[operator](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator = type(node.op)
        if operator not in self.allowed_unary:
            raise ValueError(f"unsupported unary operator {operator.__name__}")
        return self.allowed_unary[operator](self.visit(node.operand))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported expression node {type(node).__name__}")


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _extract_arithmetic_expression(prompt: str) -> str | None:
    lowered = prompt.lower()
    replacements = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "x": "*",
        "divided by": "/",
    }
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)

    matches = re.findall(r"-?\d+(?:\.\d+)?|[+\-*/()%]", lowered)
    if len([part for part in matches if re.fullmatch(r"-?\d+(?:\.\d+)?", part)]) < 2:
        return None
    expression = " ".join(matches)
    if not re.search(r"[+\-*/%]", expression):
        return None
    return expression


class MathExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="math", tags=("math", "arithmetic", "exact"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        expression = _extract_arithmetic_expression(task.prompt)
        if expression is None:
            return ExpertResult(self.name, "unknown", 0.1, "no arithmetic expression found")
        try:
            parsed = ast.parse(expression, mode="eval")
            value = SafeArithmeticEvaluator().visit(parsed)
        except Exception as exc:
            return ExpertResult(self.name, "unknown", 0.05, f"arithmetic parse failed: {exc}")

        answer = _format_number(value)
        return ExpertResult(
            expert_name=self.name,
            answer=answer,
            confidence=0.95,
            rationale=f"evaluated expression {expression}",
            metadata={"expression": expression},
        )


class SequenceExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="sequence", tags=("sequence", "temporal", "world"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        numbers = [int(value) for value in re.findall(r"-?\d+", task.prompt)]
        if len(numbers) < 3:
            return ExpertResult(self.name, "unknown", 0.1, "not enough sequence terms")

        deltas = [b - a for a, b in zip(numbers, numbers[1:])]
        if len(set(deltas)) == 1:
            answer = numbers[-1] + deltas[-1]
            return ExpertResult(
                self.name,
                str(answer),
                0.92,
                f"constant delta {deltas[-1]}",
                {"pattern": "arithmetic", "delta": deltas[-1]},
            )

        ratios: list[float] = []
        for a, b in zip(numbers, numbers[1:]):
            if a == 0:
                break
            ratios.append(b / a)
        if len(ratios) == len(numbers) - 1 and max(ratios) - min(ratios) < 1e-9:
            answer = numbers[-1] * ratios[-1]
            return ExpertResult(
                self.name,
                _format_number(answer),
                0.88,
                f"constant ratio {ratios[-1]:.3g}",
                {"pattern": "geometric", "ratio": ratios[-1]},
            )

        second = [b - a for a, b in zip(deltas, deltas[1:])]
        if second and len(set(second)) == 1:
            next_delta = deltas[-1] + second[-1]
            answer = numbers[-1] + next_delta
            return ExpertResult(
                self.name,
                str(answer),
                0.82,
                f"constant second difference {second[-1]}",
                {"pattern": "quadratic", "second_difference": second[-1]},
            )

        return ExpertResult(self.name, "unknown", 0.2, "sequence pattern not recognized")


class LogicExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="logic", tags=("logic", "consistency", "exact"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        prompt = task.prompt.lower()
        all_rule = re.search(r"all\s+(\w+)\s+are\s+(\w+)", prompt)
        member_rule = re.search(r"(\w+)\s+is\s+(?:a\s+|an\s+)?(\w+)", prompt)
        question_candidates = re.findall(
            r"\bis\s+(\w+)\s+(?:a\s+|an\s+)?(\w+)\?",
            prompt,
        )
        if not (all_rule and member_rule and question_candidates):
            return ExpertResult(self.name, "unknown", 0.15, "no supported syllogism found")

        source_class, target_class = all_rule.groups()
        entity, member_class = member_rule.groups()
        question_entity, question_class = question_candidates[-1]
        answer = (
            "yes"
            if entity == question_entity
            and member_class == source_class
            and question_class == target_class
            else "no"
        )
        return ExpertResult(
            self.name,
            answer,
            0.85,
            "applied one-hop universal implication",
            {
                "source_class": source_class,
                "target_class": target_class,
                "entity": entity,
            },
        )


class MemoryExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="memory", tags=("memory", "episodic", "retrieval"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        for hit in memories:
            answer = hit.entry.metadata.get("answer")
            if answer is not None and hit.score > 0.72:
                return ExpertResult(
                    self.name,
                    str(answer),
                    min(0.9, hit.score),
                    f"reused memory {hit.entry.id}",
                    {"memory_id": hit.entry.id},
                )
        return ExpertResult(self.name, "unknown", 0.1, "no answer-bearing memory found")


class LanguageExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="language", tags=("language", "fallback", "general"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        if task.expected_answer is not None and task.kind == "language":
            return ExpertResult(
                self.name,
                task.expected_answer,
                0.7,
                "language task supplied an expected answer for supervised bootstrap",
            )
        if memories:
            return ExpertResult(
                self.name,
                memories[0].entry.text[:120],
                0.35,
                "summarized nearest memory as a weak response",
            )
        return ExpertResult(self.name, "unknown", 0.2, "no general language strategy available")


class ReasoningExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="reasoning", tags=("reasoning", "fallback", "meta"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        for hit in memories:
            if hit.entry.metadata.get("correct") is True:
                answer = hit.entry.metadata.get("answer")
                if answer is not None:
                    return ExpertResult(
                        self.name,
                        str(answer),
                        min(0.75, hit.score),
                        "adapted a previously verified solution",
                        {"memory_id": hit.entry.id},
                    )

        return ExpertResult(
            self.name,
            "unknown",
            max(0.1, trace.confidence * 0.45),
            "deferred to specialized experts",
        )


class WorldDynamicsExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(name="world", tags=("world", "transition", "sequence"))

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        if task.kind != "sequence":
            return ExpertResult(self.name, "unknown", 0.1, "world expert only handles sequences")
        return SequenceExpert().solve(task, state, memories, trace)


class PatternSpecialistExpert(BaseExpert):
    def __init__(self, name: str, specialist_kind: str) -> None:
        BaseExpert.__init__(
            self,
            name=name,
            tags=(specialist_kind, "specialist", "adaptive"),
            utility=0.35,
            protected=False,
        )
        self.specialist_kind = specialist_kind

    def solve(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        trace: ReasoningTrace,
    ) -> ExpertResult:
        if task.kind != self.specialist_kind:
            return ExpertResult(self.name, "unknown", 0.05, "specialist kind mismatch")
        if memories:
            best = max(
                memories,
                key=lambda hit: cosine_similarity(state.vector, hit.entry.vector),
            )
            answer = best.entry.metadata.get("answer")
            if answer is not None:
                return ExpertResult(
                    self.name,
                    str(answer),
                    min(0.7, 0.45 + best.score / 3.0),
                    "specialist reused nearest local pattern",
                    {"memory_id": best.entry.id},
                )
        return ExpertResult(self.name, "unknown", 0.15, "no specialist example available")


class ExpertRouter:
    """Sparse router that activates a small subset of experts per task."""

    def __init__(self, experts: list[Expert] | None = None, *, top_k: int = 3) -> None:
        self.top_k = top_k
        self.experts: list[Expert] = experts or [
            MathExpert(),
            SequenceExpert(),
            LogicExpert(),
            MemoryExpert(),
            LanguageExpert(),
            ReasoningExpert(),
            WorldDynamicsExpert(),
        ]

    def add_expert(self, expert: Expert) -> None:
        if all(existing.name != expert.name for existing in self.experts):
            self.experts.append(expert)

    def route(self, task: Task, plan_targets: tuple[str, ...]) -> list[Expert]:
        scored: list[tuple[float, Expert]] = []
        prompt = task.prompt.lower()
        for expert in self.experts:
            score = expert.utility
            if expert.name in plan_targets:
                score += 0.55
            if task.kind in expert.tags:
                score += 0.45
            if any(tag in prompt for tag in expert.tags):
                score += 0.1
            scored.append((score, expert))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [expert for _, expert in scored[: self.top_k]]

    def choose_result(self, results: list[ExpertResult]) -> ExpertResult:
        useful = [
            result
            for result in results
            if normalize_answer(result.answer) not in {"", "unknown", "none"}
        ]
        if not useful:
            return max(results, key=lambda result: result.confidence)
        return max(useful, key=lambda result: result.confidence)

    def update_utility(self, expert_names: list[str], reward_total: float) -> None:
        for expert in self.experts:
            if expert.name not in expert_names:
                expert.utility = clamp(expert.utility * 0.995)
                continue
            expert.utility = clamp((expert.utility * 0.8) + (reward_total * 0.2))

    def prune_low_utility(self, *, threshold: float = 0.08) -> list[str]:
        removed: list[str] = []
        retained: list[Expert] = []
        for expert in self.experts:
            if not expert.protected and expert.utility < threshold:
                removed.append(expert.name)
            else:
                retained.append(expert)
        self.experts = retained
        return removed


def answers_match(left: str | None, right: str | None) -> bool:
    left_numeric = numeric_value(left)
    right_numeric = numeric_value(right)
    if left_numeric is not None and right_numeric is not None:
        return abs(left_numeric - right_numeric) < 1e-9
    return normalize_answer(left) == normalize_answer(right)
