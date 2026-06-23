from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from awr.encoder import ObservationEncoder
from awr.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_retrieve_similar_memory(self) -> None:
        encoder = ObservationEncoder()
        memory = MemoryStore(encoder)
        entry = memory.add(
            "long_term",
            "What is 2 + 2? -> 4",
            metadata={"answer": "4", "correct": True},
        )

        hits = memory.retrieve("Please solve 2 + 2", kind="long_term", top_k=1)

        self.assertEqual(hits[0].entry.id, entry.id)
        self.assertGreater(hits[0].score, 0.1)

    def test_save_and_load_memory(self) -> None:
        encoder = ObservationEncoder()
        memory = MemoryStore(encoder)
        memory.add("episodic", "Problem: A\nAnswer: B", metadata={"answer": "B"})

        path = ROOT / ".tmp-memory-test.json"
        try:
            if path.exists():
                path.unlink()
            memory.save(path)
            loaded = MemoryStore.load(path, encoder)
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(len(loaded.entries), 1)
        self.assertEqual(loaded.entries[0].metadata["answer"], "B")


if __name__ == "__main__":
    unittest.main()
