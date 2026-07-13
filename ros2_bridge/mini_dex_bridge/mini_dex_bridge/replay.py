import json
import time
from pathlib import Path

import rclpy
from std_msgs.msg import String

from mini_dex_bridge.transforms import build_command

SEQUENCE_PATH = Path("/workspace/perception/artifacts/capture_sequence.json")
# Expected format: either a JSON array of frame dicts, or a dict with {"frames": [...]}
# Each frame dict should match shared/schema/capture_schema.json
FALLBACK_PATH = Path("/workspace/shared/fixtures/capture_sample.json")


def load_frames() -> list[dict]:
    """Load capture sequence frames, handling both array and wrapped-object shapes.

    Expected inputs:
    - Plain array: [frame1, frame2, ...]
    - Wrapped object: {"frames": [frame1, frame2, ...]}

    Each frame should match shared/schema/capture_schema.json.
    """
    if SEQUENCE_PATH.exists():
        data = json.loads(SEQUENCE_PATH.read_text())
        # Handle both plain array and {"frames": [...]} wrapper
        return data["frames"] if isinstance(data, dict) else data
    fixture = json.loads(FALLBACK_PATH.read_text())
    return [fixture]


def main():
    rclpy.init()
    node = rclpy.create_node("replay_node")
    capture_pub = node.create_publisher(String, "/mini_dex/capture", 10)
    command_pub = node.create_publisher(String, "/mini_dex/joint_command", 10)

    for frame in load_frames():
        capture_msg = String()
        capture_msg.data = json.dumps(frame)
        capture_pub.publish(capture_msg)

        command = build_command(frame, num_joints=24)
        command_msg = String()
        command_msg.data = json.dumps(command)
        command_pub.publish(command_msg)

        node.get_logger().info(f"replayed frame {frame.get('frame_id')} -> command step {command['step_id']}")
        time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
