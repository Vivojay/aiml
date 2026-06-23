from __future__ import annotations

from dataclasses import dataclass, field

from awr.types import LatentState
from awr.utils import blend_vectors, clamp, cosine_similarity


@dataclass(slots=True)
class TransitionPrototype:
    key: str
    vector: tuple[float, ...]
    symbols: frozenset[str]
    observations: int = 1
    average_reward: float = 0.0
    errors: list[float] = field(default_factory=list)


class WorldModel:
    """Online latent transition model for predicting future state prototypes."""

    def __init__(self, *, learning_rate: float = 0.25, error_window: int = 50) -> None:
        self.learning_rate = learning_rate
        self.error_window = error_window
        self._transitions: dict[str, TransitionPrototype] = {}
        self.global_errors: list[float] = []

    @property
    def transitions(self) -> tuple[TransitionPrototype, ...]:
        return tuple(self._transitions.values())

    def transition_key(self, state: LatentState, action_hint: str | None = None) -> str:
        symbols = sorted(state.symbols)
        state_signature = ",".join(symbols[:8])
        return f"{state_signature}::{action_hint or 'observe'}"

    def predict_future(
        self,
        state: LatentState,
        *,
        action_hint: str | None = None,
    ) -> LatentState:
        key = self.transition_key(state, action_hint)
        prototype = self._transitions.get(key)
        if prototype is None:
            return LatentState(
                vector=state.vector,
                symbols=state.symbols,
                source_text=f"predicted-persistence:{action_hint or 'observe'}",
                confidence=0.25 * state.confidence,
            )

        confidence = clamp(0.35 + min(prototype.observations, 20) / 30.0)
        return LatentState(
            vector=prototype.vector,
            symbols=prototype.symbols,
            source_text=f"predicted-transition:{key}",
            confidence=confidence,
        )

    def observe_transition(
        self,
        current: LatentState,
        action_hint: str,
        observed_next: LatentState,
        reward: float,
    ) -> float:
        key = self.transition_key(current, action_hint)
        predicted = self.predict_future(current, action_hint=action_hint)
        error = clamp(1.0 - max(0.0, cosine_similarity(predicted.vector, observed_next.vector)))
        prototype = self._transitions.get(key)

        if prototype is None:
            self._transitions[key] = TransitionPrototype(
                key=key,
                vector=observed_next.vector,
                symbols=observed_next.symbols,
                average_reward=reward,
                errors=[error],
            )
        else:
            prototype.vector = blend_vectors(
                prototype.vector,
                observed_next.vector,
                self.learning_rate,
            )
            prototype.symbols = frozenset(set(prototype.symbols) | set(observed_next.symbols))
            prototype.observations += 1
            prototype.average_reward = (
                (prototype.average_reward * (prototype.observations - 1)) + reward
            ) / prototype.observations
            prototype.errors.append(error)
            prototype.errors = prototype.errors[-self.error_window :]

        self.global_errors.append(error)
        self.global_errors = self.global_errors[-self.error_window :]
        return error

    def mean_recent_error(self) -> float:
        if not self.global_errors:
            return 0.0
        return sum(self.global_errors) / len(self.global_errors)

    def needs_capacity(self, *, threshold: float = 0.55, min_samples: int = 8) -> bool:
        return len(self.global_errors) >= min_samples and self.mean_recent_error() > threshold
