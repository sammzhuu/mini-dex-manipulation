import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_dex_bridge"))

from transforms import build_command


def test_build_command_includes_source_object_id():
    frame = {
        "frame_id": 3,
        "timestamp": 0.1,
        "camera": {"width": 128, "height": 128, "fov_deg": 45},
        "detections": [
            {"object_id": "target_object", "position": [0.1, 0.2, 0.3], "orientation": [0, 0, 0, 1], "confidence": 0.9}
        ],
    }
    command = build_command(frame, num_joints=24)
    assert command["step_id"] == 3
    assert command["source_object_id"] == "target_object"
    assert len(command["joint_targets"]) == 24


def test_build_command_handles_no_detections():
    frame = {"frame_id": 0, "timestamp": 0.0, "camera": {"width": 128, "height": 128, "fov_deg": 45}, "detections": []}
    command = build_command(frame, num_joints=24)
    assert command["source_object_id"] is None
