"""Relay the D555's colour, aligned depth and camera_info to the *_sync topics SLAM reads.

Stamps pass through. With the D555's per-frame metadata reaching librealsense (the
uvcvideo D4XX entry in docs/D555_IMU_KERNEL_FIX.md), its stamps are on global time,
within a frame of host time, and a frame's image and camera_info share one stamp, so
depth_to_pointcloud's exact sync pairs them and RTAB-Map's approximate sync matches
them with odometry. On the sim lanes the stamps are the bridge's /clock.

``restamp:=true`` replaces each stamp with the ROS clock at reception, which gives an
image and its camera_info different stamps and breaks exact synchronizers.

Subscriptions (from RealSense):
    /d555/color/image_raw
    /d555/aligned_depth_to_color/image_raw
    /d555/color/camera_info
    /d555/aligned_depth_to_color/camera_info

Publications:
    /d555/color/image_sync
    /d555/aligned_depth_to_color/image_sync
    /d555/color/camera_info_sync
    /d555/aligned_depth_to_color/camera_info_sync
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo


class TimestampFixer(Node):

    def __init__(self):
        super().__init__("timestamp_fixer")

        self.declare_parameter("restamp", False)
        self._restamp = self.get_parameter("restamp").value

        self._first_logged = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        pairs = [
            ("/d555/color/image_raw", "/d555/color/image_sync", Image),
            (
                "/d555/aligned_depth_to_color/image_raw",
                "/d555/aligned_depth_to_color/image_sync",
                Image,
            ),
            ("/d555/color/camera_info", "/d555/color/camera_info_sync", CameraInfo),
            (
                "/d555/aligned_depth_to_color/camera_info",
                "/d555/aligned_depth_to_color/camera_info_sync",
                CameraInfo,
            ),
        ]

        self._pubs: dict[str, rclpy.publisher.Publisher] = {}
        for in_topic, out_topic, msg_type in pairs:
            pub = self.create_publisher(msg_type, out_topic, qos)
            self._pubs[in_topic] = pub
            self.create_subscription(
                msg_type,
                in_topic,
                lambda msg, p=pub: self._relay(msg, p),
                qos,
            )

        mode = "restamp" if self._restamp else "passthrough"
        self.get_logger().info(
            f"TimestampFixer ready ({mode}) — waiting for first frame"
        )

    def _relay(self, msg, pub):
        if not self._restamp:
            if not self._first_logged:
                self._first_logged = True
                hw_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.get_logger().info(
                    f"First frame relayed unchanged: stamp={hw_s:.3f}"
                )
            pub.publish(msg)
            return

        now = self.get_clock().now().to_msg()

        if not self._first_logged:
            self._first_logged = True
            hw_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            sys_s = now.sec + now.nanosec / 1e9
            self.get_logger().info(
                f"First frame re-stamped: hw={hw_s:.3f} → sys={sys_s:.3f} "
                f"(delta={sys_s - hw_s:.3f}s)"
            )

        msg.header.stamp = now
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TimestampFixer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
