from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from awr.encoder import ObservationEncoder
from awr.types import LatentState, MemoryKind
from awr.utils import clamp, cosine_similarity


@dataclass(slots=True)
class MemoryEntry:
    id: str
    kind: MemoryKind
    text: str
    vector: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    use_count: int = 0

    def reinforce(self, amount: float) -> None:
        self.strength = clamp(self.strength + amount, 0.0, 3.0)
        self.updated_at = time.time()
        self.use_count += 1


@dataclass(slots=True)
class MemoryHit:
    entry: MemoryEntry
    score: float


class MemoryStore:
    """Similarity-retrieved short-term, long-term, and episodic memory."""

    def __init__(
        self,
        encoder: ObservationEncoder,
        *,
        max_short_term: int = 1_000,
        max_long_term: int = 10_000,
        max_episodic: int = 10_000,
    ) -> None:
        self.encoder = encoder
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.max_episodic = max_episodic
        self._entries: dict[str, MemoryEntry] = {}

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries.values())

    def add(
        self,
        kind: MemoryKind,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        strength: float = 1.0,
        entry_id: str | None = None,
    ) -> MemoryEntry:
        vector = self.encoder.encode(text).vector
        entry = MemoryEntry(
            id=entry_id or str(uuid.uuid4()),
            kind=kind,
            text=text,
            vector=vector,
            metadata=metadata or {},
            strength=strength,
        )
        self._entries[entry.id] = entry
        self.prune()
        return entry

    def retrieve(
        self,
        query: LatentState | str,
        *,
        top_k: int = 5,
        kind: MemoryKind | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        state = self.encoder.encode(query) if isinstance(query, str) else query
        hits: list[MemoryHit] = []
        for entry in self._entries.values():
            if kind is not None and entry.kind != kind:
                continue
            similarity = cosine_similarity(state.vector, entry.vector)
            score = similarity * (0.5 + min(entry.strength, 2.0) / 2.0)
            if score >= min_score:
                hits.append(MemoryHit(entry=entry, score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        selected = hits[:top_k]
        for hit in selected:
            hit.entry.reinforce(0.01)
        return selected

    def reinforce(self, entry_id: str, amount: float) -> None:
        if entry_id in self._entries:
            self._entries[entry_id].reinforce(amount)

    def prune(self) -> int:
        before = len(self._entries)
        for kind, limit in (
            ("short_term", self.max_short_term),
            ("long_term", self.max_long_term),
            ("episodic", self.max_episodic),
        ):
            entries = [entry for entry in self._entries.values() if entry.kind == kind]
            if len(entries) <= limit:
                continue
            entries.sort(key=lambda entry: (entry.strength, entry.updated_at))
            for entry in entries[: len(entries) - limit]:
                del self._entries[entry.id]
        return before - len(self._entries)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {**asdict(entry), "vector": list(entry.vector)}
                for entry in self._entries.values()
            ]
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, encoder: ObservationEncoder) -> "MemoryStore":
        store = cls(encoder)
        source = Path(path)
        if not source.exists():
            return store
        data = json.loads(source.read_text(encoding="utf-8"))
        for raw in data.get("entries", []):
            entry = MemoryEntry(
                id=raw["id"],
                kind=raw["kind"],
                text=raw["text"],
                vector=tuple(float(value) for value in raw["vector"]),
                metadata=raw.get("metadata", {}),
                strength=float(raw.get("strength", 1.0)),
                created_at=float(raw.get("created_at", time.time())),
                updated_at=float(raw.get("updated_at", time.time())),
                use_count=int(raw.get("use_count", 0)),
            )
            store._entries[entry.id] = entry
        return store
