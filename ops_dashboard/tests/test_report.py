import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "shared" / "fixtures" / "rollout.json"


def test_generate_report_includes_summary_stats():
    from report import generate_report

    rollout = json.loads(FIXTURE_PATH.read_text())
    report = generate_report(rollout)
    assert rollout["task"] in report
    assert "Success rate" in report
    assert f"{rollout['summary']['num_episodes']}" in report
