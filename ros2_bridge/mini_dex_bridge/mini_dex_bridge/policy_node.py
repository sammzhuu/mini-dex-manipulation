import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mini_dex_bridge.transforms import build_command


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("num_joints", 24)
        self.subscription = self.create_subscription(String, "/mini_dex/capture", self.on_capture, 10)
        self.publisher = self.create_publisher(String, "/mini_dex/joint_command", 10)

    def on_capture(self, msg: String):
        frame = json.loads(msg.data)
        num_joints = self.get_parameter("num_joints").value
        command = build_command(frame, num_joints=num_joints)
        out = String()
        out.data = json.dumps(command)
        self.publisher.publish(out)
        self.get_logger().info(f"received frame {frame.get('frame_id')}, published joint command")


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
