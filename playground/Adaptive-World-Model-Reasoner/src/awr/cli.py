from __future__ import annotations

import argparse
import json
from pathlib import Path

from awr.agent import AdaptiveWorldModelReasoner, infer_task_kind
from awr.types import TaskKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="awr",
        description="Run the Adaptive World-Model Reasoner prototype.",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Optional JSON memory path for persistent long-term and episodic memory.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for self-play tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a deterministic mixed-task demonstration.")
    demo.add_argument("--cycles", type=int, default=9, help="Number of self-play cycles.")

    solve = subparsers.add_parser("solve", help="Solve one prompt.")
    solve.add_argument("prompt", help="Prompt to solve.")
    solve.add_argument("--answer", default=None, help="Optional expected answer for verification.")
    solve.add_argument(
        "--kind",
        choices=["arithmetic", "sequence", "logic", "language", "unknown"],
        default=None,
        help="Optional task kind; inferred when omitted.",
    )
    solve.add_argument("--difficulty", type=float, default=0.2)

    self_play = subparsers.add_parser("self-play", help="Run generated training tasks.")
    self_play.add_argument("--cycles", type=int, default=25, help="Number of generated tasks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = AdaptiveWorldModelReasoner(memory_path=args.memory, seed=args.seed)

    if args.command == "demo":
        _run_demo(agent, cycles=args.cycles)
    elif args.command == "solve":
        kind = args.kind if args.kind is not None else infer_task_kind(args.prompt)
        episode = agent.solve(
            args.prompt,
            expected_answer=args.answer,
            kind=kind,  # type: ignore[arg-type]
            difficulty=args.difficulty,
        )
        _print_episode(episode)
    elif args.command == "self-play":
        episodes = agent.self_play(args.cycles)
        for episode in episodes:
            _print_episode(episode, compact=True)
        print(json.dumps(agent.snapshot(), indent=2))
    else:
        raise ValueError(f"unknown command {args.command}")

    if args.memory is not None:
        agent.save_memory(args.memory)
    return 0


def _run_demo(agent: AdaptiveWorldModelReasoner, *, cycles: int) -> None:
    prompts: list[tuple[str, str, TaskKind]] = [
        ("What is 7 + 8?", "15", "arithmetic"),
        ("Complete the sequence: 2, 5, 8, 11, ?", "14", "sequence"),
        ("All dax are mip. Nira is a dax. Is Nira mip?", "yes", "logic"),
    ]
    for prompt, answer, kind in prompts:
        _print_episode(agent.solve(prompt, expected_answer=answer, kind=kind))
    print(f"\nSelf-play cycles: {cycles}")
    for episode in agent.self_play(cycles):
        _print_episode(episode, compact=True)
    print("\nSnapshot")
    print(json.dumps(agent.snapshot(), indent=2))


def _print_episode(episode, *, compact: bool = False) -> None:
    status = "correct" if episode.verification.correct else "incorrect"
    print(
        f"[{status}] {episode.task.id} {episode.task.kind}: "
        f"{episode.task.prompt} -> {episode.prediction.answer} "
        f"(reward={episode.reward.total:.3f})"
    )
    if compact:
        return
    if episode.verification.error_trace:
        print("  errors:")
        for error in episode.verification.error_trace:
            print(f"  - {error}")
    print(f"  chosen expert: {episode.prediction.metadata.get('chosen_expert')}")
    print(f"  trace steps: {len(episode.prediction.trace.steps)}")
    for lesson in episode.lessons:
        print(f"  lesson: {lesson}")
    for event in episode.adaptation_events:
        print(f"  adaptation: {event.action} ({event.reason})")


if __name__ == "__main__":
    raise SystemExit(main())
