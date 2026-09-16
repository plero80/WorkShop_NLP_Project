# Code

| Directory | Purpose |
|---|---|
| [core/](core/) | Shared reward, kNN, policy optimization, evaluation, and launch code |
| [experiments/](experiments/) | Best-of-N, kNN distillation, GSM8K using shared PPO, and its YAML CLI |
| [tests/](tests/) | Original tests and GSM8K integration tests |
| [templates/](templates/) | Original human-review HTML template |

## Shared implementation

| Files in `core/` | Role |
|---|---|
| `knn_core.py`, `gap_correction.py`, `reward_bridge.py` | kNN estimates and reward correction |
| `ppo_engine.py` | Policy optimization and checkpoint state |
| `run_study.py`, `improvement.py`, `evaluation.py` | Follow-up experiment, memory refresh, and evaluation |
| `assets.py`, `chat_format.py`, `common.py` | Model assets, conversation formatting, and run identities |
| `reporting.py`, `review.py` | Saved reports and human-review materials |
| `launch.py`, `setup_environment.py`, `RESUME_KNN_DISTILLATION.py` | Original setup and operational scripts |

The shared core and retained experiments are exact copies arranged for reading.
The new `gsm8k_experiment` adds dataset/scoring adapters and calls the shared
`PPOActor`, `PPOTrainer`, GAE, and cosine search. Use the [runtime restoration
instructions](../README.md#running-the-code) before running them, because their
imports and resource paths still refer to the original execution layout.
