from __future__ import annotations

import re
from pathlib import Path

from awr.continual_learning import ContinualLearner
from awr.dynamic_structure import DynamicArchitectureController
from awr.encoder import ObservationEncoder
from awr.experts import ExpertRouter
from awr.memory import MemoryStore
from awr.reasoning import FastReasoner, SlowReasoner
from awr.rl import RewardEngine
from awr.tasks import SelfGeneratedTaskSystem
from awr.types import EpisodeRecord, Observation, Prediction, Task, TaskKind
from awr.verifiers import CompositeVerifier
from awr.world_model import WorldModel


class AdaptiveWorldModelReasoner:
    """End-to-end prototype that follows the README execution pipeline."""

    def __init__(
        self,
        *,
        encoder: ObservationEncoder | None = None,
        memory: MemoryStore | None = None,
        memory_path: str | Path | None = None,
        seed: int = 7,
    ) -> None:
        self.encoder = encoder or ObservationEncoder()
        self.memory_path = Path(memory_path) if memory_path is not None else None
        if memory is not None:
            self.memory = memory
        elif self.memory_path is not None:
            self.memory = MemoryStore.load(self.memory_path, self.encoder)
        else:
            self.memory = MemoryStore(self.encoder)

        self.world_model = WorldModel()
        self.slow_reasoner = SlowReasoner()
        self.fast_reasoner = FastReasoner()
        self.router = ExpertRouter()
        self.verifier = CompositeVerifier(self.memory)
        self.reward_engine = RewardEngine()
        self.task_system = SelfGeneratedTaskSystem(seed=seed)
        self.learner = ContinualLearner(self.memory)
        self.dynamic_controller = DynamicArchitectureController()
        self.episodes: list[EpisodeRecord] = []

    def run_task(self, task: Task) -> EpisodeRecord:
        observation = Observation(task.prompt, metadata={"task_id": task.id, "kind": task.kind})
        current_state = self.encoder.encode_observation(observation)
        memories = self.memory.retrieve(current_state, top_k=5, min_score=0.12)
        predicted_future = self.world_model.predict_future(current_state, action_hint=task.kind)

        plan = self.slow_reasoner.plan(task, current_state, memories)
        trace = self.fast_reasoner.run(task, current_state, memories, plan)

        selected_experts = self.router.route(task, plan.target_experts)
        trace.selected_experts = [expert.name for expert in selected_experts]
        expert_results = [
            expert.solve(task, current_state, memories, trace)
            for expert in selected_experts
        ]
        for result in expert_results:
            trace.steps.append(
                f"{result.expert_name} -> {result.answer!r} "
                f"(confidence={result.confidence:.2f}; {result.rationale})"
            )

        chosen = self.router.choose_result(expert_results)
        answer_state = self.encoder.encode(
            f"{task.prompt} -> {chosen.answer}",
            confidence=chosen.confidence,
        )
        prediction = Prediction(
            answer=chosen.answer,
            confidence=chosen.confidence,
            latent_state=answer_state,
            trace=trace,
            metadata={
                "chosen_expert": chosen.expert_name,
                "expert_results": [
                    {
                        "expert": result.expert_name,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "rationale": result.rationale,
                    }
                    for result in expert_results
                ],
                "predicted_future_confidence": predicted_future.confidence,
            },
        )

        verification = self.verifier.verify(task, prediction)
        reward = self.reward_engine.compute(task, prediction, verification)
        self.router.update_utility(trace.selected_experts, reward.total)

        episode = EpisodeRecord(
            task=task,
            prediction=prediction,
            verification=verification,
            reward=reward,
        )
        episode.lessons.extend(self.learner.learn_from_episode(episode))

        observed_answer = task.expected_answer if task.expected_answer is not None else chosen.answer
        observed_next = self.encoder.encode(
            f"{task.prompt} -> {observed_answer}; correct={verification.correct}",
            confidence=verification.confidence,
        )
        transition_error = self.world_model.observe_transition(
            current_state,
            task.kind,
            observed_next,
            reward.total,
        )
        prediction.metadata["world_model_error"] = transition_error
        trace.steps.append(f"world model transition error={transition_error:.3f}")

        self.task_system.record_result(task, verification)
        episode.adaptation_events.extend(
            self.dynamic_controller.adapt(
                episode,
                router=self.router,
                memory=self.memory,
                world_model=self.world_model,
            )
        )

        self.episodes.append(episode)
        if self.memory_path is not None:
            self.memory.save(self.memory_path)
        return episode

    def solve(
        self,
        prompt: str,
        *,
        expected_answer: str | None = None,
        kind: TaskKind | None = None,
        difficulty: float = 0.2,
    ) -> EpisodeRecord:
        task_kind = kind or infer_task_kind(prompt)
        task = Task(
            id=f"user-{len(self.episodes) + 1:05d}",
            kind=task_kind,
            prompt=prompt,
            expected_answer=expected_answer,
            difficulty=difficulty,
        )
        return self.run_task(task)

    def self_play(self, cycles: int) -> list[EpisodeRecord]:
        episodes: list[EpisodeRecord] = []
        for _ in range(cycles):
            task = self.task_system.generate()
            episodes.append(self.run_task(task))
        return episodes

    def save_memory(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path is not None else self.memory_path
        if destination is None:
            raise ValueError("no memory path configured")
        self.memory.save(destination)

    def snapshot(self) -> dict[str, object]:
        return {
            "episodes": len(self.episodes),
            "memory_entries": len(self.memory.entries),
            "experts": [expert.name for expert in self.router.experts],
            "world_transitions": len(self.world_model.transitions),
            "recent_world_error": self.world_model.mean_recent_error(),
            "task_stats": {
                kind: {
                    "attempts": stats.attempts,
                    "successes": stats.successes,
                    "accuracy": stats.accuracy,
                    "average_confidence": stats.average_confidence,
                }
                for kind, stats in self.task_system.stats.items()
            },
        }


def infer_task_kind(prompt: str) -> TaskKind:
    lowered = prompt.lower()
    if "sequence" in lowered or re.search(r"\d+\s*,\s*\d+\s*,\s*\d+", lowered):
        return "sequence"
    if "all " in lowered and " are " in lowered and lowered.strip().endswith("?"):
        return "logic"
    if re.search(r"\d+\s*[+\-*/]\s*\d+", lowered):
        return "arithmetic"
    if any(word in lowered for word in ("plus", "minus", "times", "divided by")):
        return "arithmetic"
    return "unknown"
