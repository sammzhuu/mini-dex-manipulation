import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_load_rollout_returns_valid_fixture():
    from data import DEFAULT_FIXTURE, load_rollout

    rollout = load_rollout(DEFAULT_FIXTURE)
    assert rollout["task"]
    assert 0.0 <= rollout["summary"]["success_rate"] <= 1.0


def test_load_rollout_rejects_malformed_data(tmp_path):
    from data import load_rollout

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"task": "x"}))  # missing required fields
    with pytest.raises(jsonschema.ValidationError):
        load_rollout(bad_path)
