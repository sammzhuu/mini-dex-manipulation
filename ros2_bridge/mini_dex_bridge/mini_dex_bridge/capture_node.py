import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DEFAULT_FIXTURE = Path("/workspace/shared/fixtures/capture_sample.json")


class CaptureNode(Node):
    def __init__(self):
        super().__init__("capture_node")
        self.declare_parameter("fixture_path", str(DEFAULT_FIXTURE))
        self.declare_parameter("rate_hz", 5.0)
        self.publisher = self.create_publisher(String, "/mini_dex/capture", 10)
        self.base_frame = json.loads(Path(self.get_parameter("fixture_path").value).read_text())
        self.frame_id = 0
        rate_hz = self.get_parameter("rate_hz").value
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)

    def tick(self):
        payload = dict(self.base_frame)
        payload["frame_id"] = self.frame_id
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)
        self.get_logger().info(f"published capture frame {self.frame_id}")
        self.frame_id += 1


def main():
    rclpy.init()
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
