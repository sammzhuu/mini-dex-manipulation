import json
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "shared" / "schema" / "rollout_schema.json"
DEFAULT_FIXTURE = REPO_ROOT / "shared" / "fixtures" / "rollout.json"


def load_rollout(path: Path = DEFAULT_FIXTURE) -> dict:
    schema = json.loads(SCHEMA_PATH.read_text())
    data = json.loads(Path(path).read_text())
    jsonschema.validate(instance=data, schema=schema)
    return data
