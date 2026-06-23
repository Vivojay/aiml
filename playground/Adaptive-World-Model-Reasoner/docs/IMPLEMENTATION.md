# Implementation Map

This prototype turns the README architecture into a runnable, dependency-light
Python system. It is intentionally small, but every major component has a real
interface and participates in the execution loop.

## Component Mapping

| README component | Implementation |
| --- | --- |
| World model | `awr.world_model.WorldModel` predicts and updates latent transition prototypes. |
| Hierarchical reasoning | `awr.reasoning.SlowReasoner` plans; `awr.reasoning.FastReasoner` runs local internal cycles. |
| Persistent memory | `awr.memory.MemoryStore` stores short-term, long-term, and episodic memories with similarity retrieval and JSON persistence. |
| Self-generated tasks | `awr.tasks.SelfGeneratedTaskSystem` generates arithmetic, sequence, and logic tasks with adaptive difficulty. |
| Verifier | `awr.verifiers.CompositeVerifier` combines exact answer checks with memory consistency checks. |
| Reinforcement learning | `awr.rl.RewardEngine` converts correctness, novelty, efficiency, consistency, and difficulty into reward. |
| Dynamic growth | `awr.dynamic_structure.DynamicArchitectureController` adds adaptive specialists under sustained error. |
| Dynamic compression | The same controller prunes low-utility adaptive experts and enforces memory limits. |
| Sparse experts | `awr.experts.ExpertRouter` activates a small expert subset per task. |
| Continual learning | `awr.continual_learning.ContinualLearner` records episodes, learns beliefs, and supports unlearning. |

## Running

From the repository root:

```powershell
python run_awr.py demo --cycles 9
python run_awr.py solve "What is 12 * 7?" --answer 84
python run_awr.py self-play --cycles 25 --memory .awr/memory.json
```

The `--memory` option persists long-term and episodic memory as JSON.

## Design Notes

- The latent encoder is a hashing encoder, not a trained neural model.
- Experts are symbolic first-pass stand-ins for future learned modules.
- Growth and pruning are represented as explicit runtime events so future neural
  modules can attach at the same boundary.
- The system is testable without downloading model weights or dependencies.
