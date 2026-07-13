import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "shared" / "schema" / "rollout_schema.json"
FIXTURE_PATH = REPO_ROOT / "shared" / "fixtures" / "rollout.json"


def test_fixture_matches_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    fixture = json.loads(FIXTURE_PATH.read_text())
    jsonschema.validate(instance=fixture, schema=schema)


def test_generated_rollout_matches_schema():
    generated_path = REPO_ROOT / "sim_policy" / "artifacts" / "rollout.json"
    if not generated_path.exists():
        pytest.skip("run evaluate.py first to generate artifacts/rollout.json")
    schema = json.loads(SCHEMA_PATH.read_text())
    generated = json.loads(generated_path.read_text())
    jsonschema.validate(instance=generated, schema=schema)
