# Matched HH-RLHF M2 continuation

**RIDGE PPO NOT RUN. These are the verified existing controls only.**

Start: update 300; endpoint: update 400. Seeds: 42, 43, 44. Shared final prompts: 512.

| seed | branch | answers | mean_judge_z | mean_proxy_z | high_gap_rate | mean_response_tokens | completion_eos_rate | length_capped_rate | refusal_diagnostic_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | proxy | 512 | -0.2119 | -0.2617 | 0.0332 | 138.0000 | 0.7793 | 0.2207 | 0.2812 |
| 42 | knn | 512 | -0.2076 | -0.3010 | 0.0352 | 122.9199 | 0.8262 | 0.1738 | 0.3418 |
| 43 | proxy | 512 | -0.1877 | -0.2238 | 0.0332 | 144.1074 | 0.7500 | 0.2500 | 0.2949 |
| 43 | knn | 512 | -0.1786 | -0.2691 | 0.0371 | 118.9922 | 0.8262 | 0.1738 | 0.3613 |
| 44 | proxy | 512 | -0.1920 | -0.2248 | 0.0488 | 136.7422 | 0.7734 | 0.2266 | 0.2930 |
| 44 | knn | 512 | -0.2252 | -0.3080 | 0.0410 | 127.7324 | 0.8047 | 0.1953 | 0.3535 |

Seed means and sample standard deviations are in `policy_seed_summary.csv`.

Completion is the EOS fraction, not a semantic completeness rating. Refusal is a fixed phrase heuristic over the first 300 answer characters, not a measure of inappropriate refusals. Inspect answers or complete a blinded human review before making those claims.

Controls were checked for parent checkpoint, policy/value/optimizer/RNG fingerprint, exact training prompt schedule, update count, source code, scoring protocol, and evaluation prompts. The ridge arm uses the unchanged PPO trainer.
