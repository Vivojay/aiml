from __future__ import annotations

from awr.memory import MemoryHit
from awr.types import LatentState, Plan, ReasoningTrace, Task


class SlowReasoner:
    """Updates sparse high-level strategy and expert targets."""

    def __init__(self, *, update_period: int = 3) -> None:
        self.update_period = update_period
        self.cycles = 0
        self.last_plan: Plan | None = None

    def plan(self, task: Task, state: LatentState, memories: list[MemoryHit]) -> Plan:
        self.cycles += 1
        if self.last_plan is not None and self.cycles % self.update_period != 1:
            return self.last_plan

        notes: list[str] = []
        target_experts: tuple[str, ...]
        if memories and memories[0].score > 0.82:
            notes.append(f"high-similarity memory:{memories[0].entry.id}")

        if task.kind == "arithmetic":
            strategy = "parse exact arithmetic, then verify numerically"
            target_experts = ("math", "reasoning")
        elif task.kind == "sequence":
            strategy = "infer temporal pattern before answering"
            target_experts = ("sequence", "world", "reasoning")
        elif task.kind == "logic":
            strategy = "normalize premises and test conclusion consistency"
            target_experts = ("logic", "reasoning")
        elif memories:
            strategy = "retrieve similar solved episodes and adapt"
            target_experts = ("memory", "language", "reasoning")
        else:
            strategy = "decompose prompt and use general reasoning"
            target_experts = ("reasoning", "language")

        internal_cycles = 2 + int(task.difficulty * 4.0) + (1 if memories else 0)
        self.last_plan = Plan(
            strategy=strategy,
            target_experts=target_experts,
            internal_cycles=max(2, min(8, internal_cycles)),
            notes=tuple(notes),
        )
        return self.last_plan


class FastReasoner:
    """Performs local iterative reasoning against the current plan."""

    def run(
        self,
        task: Task,
        state: LatentState,
        memories: list[MemoryHit],
        plan: Plan,
    ) -> ReasoningTrace:
        trace = ReasoningTrace(plan=plan)
        trace.steps.append(f"encoded observation with {len(state.symbols)} symbols")
        if memories:
            trace.selected_memory_ids.extend(hit.entry.id for hit in memories)
            top = memories[0]
            trace.steps.append(
                f"retrieved {len(memories)} memories; top score={top.score:.3f}"
            )
        else:
            trace.steps.append("no relevant memories found")

        for cycle in range(plan.internal_cycles):
            if cycle == 0:
                trace.steps.append(f"strategy: {plan.strategy}")
            elif cycle == plan.internal_cycles - 1:
                trace.steps.append("final consistency pass before expert action")
            else:
                trace.steps.append(f"internal refinement cycle {cycle}")

        memory_bonus = min(0.2, len(memories) * 0.04)
        trace.confidence = min(1.0, 0.45 + task.difficulty * 0.1 + memory_bonus)
        return trace
