from __future__ import annotations

from awr.types import LatentState, Observation
from awr.utils import l2_normalize, stable_hash, tokenize


class ObservationEncoder:
    """Hashing encoder that turns text observations into compact latent states."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def encode(self, text: str, confidence: float | None = None) -> LatentState:
        tokens = tokenize(text)
        vector = [0.0] * self.dimensions
        for token in tokens:
            hashed = stable_hash(token)
            index = hashed % self.dimensions
            sign = 1.0 if (hashed >> 1) & 1 else -1.0
            vector[index] += sign

        normalized = l2_normalize(vector)
        if confidence is None:
            confidence = min(1.0, 0.2 + len(set(tokens)) / 20.0)
        return LatentState(
            vector=normalized,
            symbols=frozenset(tokens),
            source_text=text,
            confidence=confidence,
        )

    def encode_observation(self, observation: Observation) -> LatentState:
        return self.encode(observation.text)
