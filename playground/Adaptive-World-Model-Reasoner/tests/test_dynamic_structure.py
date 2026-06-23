from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from awr.agent import AdaptiveWorldModelReasoner
from awr.types import Task


class DynamicStructureTests(unittest.TestCase):
    def test_growth_event_after_repeated_failures(self) -> None:
        agent = AdaptiveWorldModelReasoner(seed=5)
        events = []
        for index in range(4):
            task = Task(
                id=f"bad-{index}",
                kind="unknown",
                prompt=f"Unsolvable synthetic task {index}",
                expected_answer="not-the-fallback",
                difficulty=0.9,
            )
            episode = agent.run_task(task)
            events.extend(episode.adaptation_events)

        self.assertTrue(any(event.action == "grow_expert" for event in events))
        self.assertTrue(any("unknown-specialist" in expert for expert in agent.snapshot()["experts"]))


if __name__ == "__main__":
    unittest.main()
