from __future__ import annotations

from collections import defaultdict, deque

from awr.experts import ExpertRouter, PatternSpecialistExpert
from awr.memory import MemoryStore
from awr.types import AdaptationEvent, EpisodeRecord
from awr.world_model import WorldModel


class DynamicArchitectureController:
    """Adds specialists under sustained error and prunes unused adaptive pieces."""

    def __init__(
        self,
        *,
        window: int = 12,
        growth_error_rate: float = 0.45,
        min_kind_samples: int = 4,
    ) -> None:
        self.window = window
        self.growth_error_rate = growth_error_rate
        self.min_kind_samples = min_kind_samples
        self.recent_by_kind: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self.growth_counts: dict[str, int] = defaultdict(int)

    def adapt(
        self,
        episode: EpisodeRecord,
        *,
        router: ExpertRouter,
        memory: MemoryStore,
        world_model: WorldModel,
    ) -> list[AdaptationEvent]:
        events: list[AdaptationEvent] = []
        kind = episode.task.kind
        self.recent_by_kind[kind].append(episode.verification.correct)

        results = self.recent_by_kind[kind]
        if len(results) >= self.min_kind_samples:
            error_rate = 1.0 - (sum(1 for value in results if value) / len(results))
            already_specialized = any(
                "adaptive" in expert.tags and kind in expert.tags for expert in router.experts
            )
            if error_rate >= self.growth_error_rate and not already_specialized:
                self.growth_counts[kind] += 1
                name = f"{kind}-specialist-{self.growth_counts[kind]}"
                router.add_expert(PatternSpecialistExpert(name, kind))
                events.append(
                    AdaptationEvent(
                        action="grow_expert",
                        reason=f"{kind} error rate {error_rate:.2f} exceeded threshold",
                        details={"expert": name, "kind": kind},
                    )
                )

        if world_model.needs_capacity():
            old_limit = memory.max_long_term
            memory.max_long_term = int(memory.max_long_term * 1.1)
            events.append(
                AdaptationEvent(
                    action="expand_memory",
                    reason="world-model prediction error remained high",
                    details={"old_limit": old_limit, "new_limit": memory.max_long_term},
                )
            )

        removed = router.prune_low_utility()
        if removed:
            events.append(
                AdaptationEvent(
                    action="prune_experts",
                    reason="adaptive experts fell below utility threshold",
                    details={"experts": removed},
                )
            )

        pruned_memories = memory.prune()
        if pruned_memories:
            events.append(
                AdaptationEvent(
                    action="compress_memory",
                    reason="memory capacity limit enforced",
                    details={"entries_removed": pruned_memories},
                )
            )

        return events
