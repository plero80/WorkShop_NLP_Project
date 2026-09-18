"""Explicit compatibility for the released fast runner's grading retry fix."""
from pathlib import Path

from .common import digest, read_json

RECOVERY_ID = "grading_retry_v1"
RECOVERY_POLICY = {
    "id": RECOVERY_ID,
    "extra_max_new_tokens": [640, 1280, 2560],
    "trigger": "no valid grade after the unchanged primary and first retry",
    "rubric_and_parser": "unchanged",
    "existing_valid_scores": "retained",
}


FORMAT_ID = "grading_format_v1"
FORMAT_POLICY = {
    "id": FORMAT_ID,
    "trigger": "legacy parser failed on a complete, non-length-capped grader reply",
    "accepted_form": "Judgement rationale ending with one explicit sentence: [Therefore, ]the correctness score is N.",
    "range": [1, 5],
    "ambiguity": "reject additional correctness-score declarations; never infer a missing digit",
    "prompts_models_budgets_embeddings": "unchanged",
    "existing_valid_scores": "retained",
}


def validate_migration(output, identity):
    if any((Path(output) / "recovery").glob("*/migration.json")):
        raise ValueError("Standalone ZIP migrations are incompatible with the shared PPO engine. Use a fresh output directory.")
    return None


def checkpoint_parents(output, fingerprint):
    manifest = read_json(Path(output) / "manifest.json")
    if manifest["fingerprint"] != fingerprint or digest(manifest["identity"]) != fingerprint:
        raise ValueError("Current experiment identity does not match its manifest.")
    validate_migration(output, manifest["identity"])
    return ()
