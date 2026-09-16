# Preservation and runtime restoration

`file_map.json` records each retained file's original runtime path, current
organized path, byte size, and SHA-256. `excluded_removed_components` records
removed source files, notebooks, and historical artifacts; neither `restore` nor `organize` recreates
them from the original snapshot. Relocation leaves retained scientific bytes unchanged.

From `workshop_project/`:

```text
python -B reproducibility/manage.py verify --code-only
python -B reproducibility/manage.py verify
python -B reproducibility/manage.py restore --destination ../run/reward_gap_followup
```

The code-only check covers retained source, settings, notebooks, and original
inputs. Full verification also covers saved results and prerequisites.
Restoration requires a new directory and starts no experiment. A `--code-only`
restore is sufficient for offline source checks; running dependent experiments
requires their saved parent studies as well.

[Saved prerequisites](../data/prerequisites/README.md) and their cohorts
restore to the paths expected by the retained code.
The retained notebook groups are Best-of-N, exploration, distillation, and follow-up.
GSM8K is a new addition using the unchanged shared PPO engine.

`additions.json` records the new GSM8K and YAML CLI files separately from the original byte
preservation map. Both `verify` and `restore` check both maps and reject path
collisions. `organize` reconstructs only the immutable original layer.
`gsm8k_import.json` records the supplied ZIP checksum, member hashes, and adaptations.
The GSM8K [run guide](../docs/gsm8k/README.md) describes protocol differences.

`original_manifests/` contains retained original package records and source guards.
`active_scope_verification.json` records checks of the four-family selection
before GSM8K was added; `gsm8k_validation.json` records the first GSM8K integration checks.
`experiment_cli_validation.json` records the subsequent YAML CLI checks; its test
report is in `active_validation/experiment_cli_tests.xml`. These reports describe
their respective source snapshots, rather than claiming later changes were tested.
`prior_validation/`, `organization_verification.json`, and other earlier check
records describe the initial, larger project and are historical evidence.
The previous full snapshot and submission ZIPs remain in `../../original_project/`.

Saved run identities and source guards have not been rewritten. Extra root
scripts can still affect a new follow-up run's identity, as described in the
[saved-run catalog](../docs/SAVED_RUNS.md). Preservation is not a new GPU reproduction.
