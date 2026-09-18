# GSM8K matched-seed suite

Seeds: [42]. Values below are mean ± sample standard deviation across training seeds, not confidence intervals.

| Policy | Strict accuracy | Numeric matches / all | Valid box | Unresolved |
|---|---:|---:|---:|---:|
| Base policy (no PPO) | 28.81% (one seed) | 29.19% (one seed) | 96.06% (one seed) | 2.27% (one seed) |
| Proxy PPO (1.5B grader) | 34.42% (one seed) | 34.42% (one seed) | 97.73% (one seed) | 2.12% (one seed) |
| Judge PPO (4B grader) | 47.69% (one seed) | 47.92% (one seed) | 96.74% (one seed) | 2.65% (one seed) |
| Static kNN PPO (4B memory) | 40.11% (one seed) | 41.70% (one seed) | 92.65% (one seed) | 3.49% (one seed) |

The data partition is held fixed across seeds. Policy initialization, sampled candidates, and PPO randomness vary by seed. Each seed independently prepares its correction memory. Inspect the per-seed results; the reported standard deviation is not a confidence interval.
