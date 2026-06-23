from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from awr.agent import AdaptiveWorldModelReasoner


class PipelineTests(unittest.TestCase):
    def test_solves_arithmetic_task(self) -> None:
        agent = AdaptiveWorldModelReasoner(seed=1)
        episode = agent.solve("What is 9 * 6?", expected_answer="54", kind="arithmetic")

        self.assertTrue(episode.verification.correct)
        self.assertEqual(episode.prediction.answer, "54")
        self.assertGreater(episode.reward.total, 0.7)

    def test_solves_sequence_task(self) -> None:
        agent = AdaptiveWorldModelReasoner(seed=1)
        episode = agent.solve(
            "Complete the sequence: 4, 7, 10, 13, ?",
            expected_answer="16",
            kind="sequence",
        )

        self.assertTrue(episode.verification.correct)
        self.assertEqual(episode.prediction.answer, "16")

    def test_solves_logic_task(self) -> None:
        agent = AdaptiveWorldModelReasoner(seed=1)
        episode = agent.solve(
            "All dax are mip. Nira is a dax. Is Nira mip?",
            expected_answer="yes",
            kind="logic",
        )

        self.assertTrue(episode.verification.correct)
        self.assertEqual(episode.prediction.answer, "yes")

    def test_self_play_records_learning_state(self) -> None:
        agent = AdaptiveWorldModelReasoner(seed=3)
        episodes = agent.self_play(12)
        snapshot = agent.snapshot()

        self.assertEqual(len(episodes), 12)
        self.assertEqual(snapshot["episodes"], 12)
        self.assertGreater(snapshot["memory_entries"], 12)
        self.assertGreaterEqual(snapshot["world_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
