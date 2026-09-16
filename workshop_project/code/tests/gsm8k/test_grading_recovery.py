import json
from types import SimpleNamespace

import numpy as np
import pytest

from gsm8k_experiment.common import atomic_json, digest
from gsm8k_experiment.memory import GapMemory, Normalization
from gsm8k_experiment.run import complete_cached_monitor, recover_cached_monitors


def test_recover_old_monitor_from_saved_answers_without_policy_generation(tmp_path):
    rows = [{"id": "a", "question": "What is 1+2?", "reference": "#### 3"},
            {"id": "b", "question": "What is 4+5?", "reference": "#### 9"}]
    answers = [{**r, "response": f"\\boxed{{{3 if i == 0 else 9}}}", "response_tokens": 3,
                "length_capped": False} for i, r in enumerate(rows)]
    config = {"generation": {"batch_size": 2}}
    folder = tmp_path / "generations" / "monitor" / "judge" / "000325"
    atomic_json(folder / "000000.json", {"input_hash": digest({"rows": rows, "repeats": 1, "greedy": True}),
                                         "items": answers})
    class NoGeneration:
        def sample(self, *a, **k):
            raise AssertionError("Must not generate answers with a different policy checkpoint")
    class Scorer:
        identity = "frozen-proxy"
        def score(self, items, stage):
            assert items == answers
            return [{"score": 5, "embedding": np.array([1., 0.], np.float32),
                     "judge_output": "Correctness_score: 5"} for _ in items]
    norm = Normalization(3, 1, 3, 1, 1)
    memory = GapMemory([[1., 0.], [0., 1.]], [0, 1], ["c", "d"], 1, .1, "frozen-proxy")
    recover_cached_monitors(NoGeneration(), rows, Scorer(), Scorer(), norm, memory, tmp_path, ["judge"], config)
    result = json.loads((tmp_path / "evaluations/monitor/judge/step_000325/metrics.json").read_text())
    assert result["accuracy"] == 1 and result["update"] == 325 and result["n"] == 2
    assert (tmp_path / "evaluations/monitor/judge/step_000325/grading_recovery.json").exists()


def test_incomplete_or_wrong_historical_cache_is_not_regenerated(tmp_path):
    rows = [{"id": "a", "question": "1+2?", "reference": "#### 3"}]
    config = {"generation": {"batch_size": 2}}
    assert complete_cached_monitor(rows, tmp_path, config) is None
    atomic_json(tmp_path / "000000.json", {"input_hash": "wrong", "items": rows})
    with pytest.raises(ValueError, match="does not match"):
        complete_cached_monitor(rows, tmp_path, config)
