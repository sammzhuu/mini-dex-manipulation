"""Pure transform logic, kept separate from rclpy node code so it can be
unit-tested without a running ROS2 graph."""


def build_command(frame: dict, num_joints: int = 24) -> dict:
    detections = frame.get("detections") or []
    source_object_id = detections[0]["object_id"] if detections else None
    return {
        "episode_id": 0,
        "step_id": frame.get("frame_id", 0),
        # Zero-vector placeholder: this lane demonstrates the message contract
        # and node wiring, not a live policy loop (see this lane's task file
        # "Scope" section) — real joint targets come from sim_policy in a
        # future integration pass.
        "joint_targets": [0.0] * num_joints,
        "source_object_id": source_object_id,
    }
