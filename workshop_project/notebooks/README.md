# Selected notebooks

| Directory | Workflows |
|---|---|
| [best_of_n/](best_of_n/) | Best-of-N development and reserved confirmation |
| [exploration/](exploration/) | Interactive reward-gap clusters and second-refresh exploration |
| [distillation/](distillation/) | kNN reward distillation |
| [followup/](followup/) | Original follow-up experiment |
| [gsm8k/](gsm8k/) | New GSM8K experiment using the shared PPO engine |

The four retained groups' cells and outputs are unchanged. The new GSM8K notebook
has no saved training results; see its [run guide](../docs/gsm8k/README.md).
The confirmation notebook is available; the saved Best-of-N result is development only.

Execute notebooks from the [restored runtime layout](../README.md#running-the-code).
Best-of-N uses saved teacher-comparison outputs; distillation and the second
explorer use saved memory-refresh outputs. Their prerequisite artifacts remain
under [data/prerequisites/](../data/prerequisites/README.md).
