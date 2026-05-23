"""Re-stamp camera messages from hardware clock to ROS system time.

The RealSense D555 on Jetson (Tegra USB) produces timestamps from its
internal hardware clock, which drifts unpredictably relative to the host
system clock (observed running ~44x faster on Tegra USB).  RTAB-Map's
approximate sync cannot match camera messages (HW time) with wheel
odometry (system time).

This node replaces every camera header stamp with the current ROS clock
time at the moment of reception.  Messages from all four camera topics
arrive within microseconds of each other per frame, so RTAB-Map's
approx_sync can match them with odom and scan data.

Subscriptions (from RealSense, HW-stamped):
    /d555/color/image_raw
    /d555/aligned_depth_to_color/image_raw
    /d555/color/camera_info
    /d555/aligned_depth_to_color/camera_info
    /d555/depth/color/points

Publications (re-stamped with system time):
    /d555/color/image_sync
    /d555/aligned_depth_to_color/image_sync
    /d555/color/camera_info_sync
    /d555/aligned_depth_to_color/camera_info_sync
    /d555/depth/color/points_sync
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


class TimestampFixer(Node):

    def __init__(self):
        super().__init__("timestamp_fixer")

        # When the upstream publisher already stamps messages on the same
        # clock as wheel odometry (e.g. the Isaac Sim ROS 2 bridge, which
        # stamps every topic with sim-time), rewriting to the Jetson's
        # system clock would de-sync cameras from odom. Expose a parameter
        # so the sim-in-the-loop launch can flip this to pass-through.
        self.declare_parameter("restamp", True)
        self._restamp = self.get_parameter("restamp").value

        self._first_logged = False

        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Per-topic (sub_qos, pub_qos). The realsense pointcloud publisher
        # defaults to SENSOR_DATA (BEST_EFFORT, depth 5); a RELIABLE sub
        # never matches it and we get an empty /scan. Publish RELIABLE so
        # downstream consumers (pointcloud_to_laserscan, rosbag) match
        # the default subscriber QoS.
        pairs = [
            ("/d555/color/image_raw",
             "/d555/color/image_sync", Image, reliable_qos, reliable_qos),
            ("/d555/aligned_depth_to_color/image_raw",
             "/d555/aligned_depth_to_color/image_sync", Image,
             reliable_qos, reliable_qos),
            ("/d555/color/camera_info",
             "/d555/color/camera_info_sync", CameraInfo,
             reliable_qos, reliable_qos),
            ("/d555/aligned_depth_to_color/camera_info",
             "/d555/aligned_depth_to_color/camera_info_sync", CameraInfo,
             reliable_qos, reliable_qos),
            ("/d555/depth/color/points",
             "/d555/depth/color/points_sync", PointCloud2,
             qos_profile_sensor_data, reliable_qos),
        ]

        self._pubs: dict[str, rclpy.publisher.Publisher] = {}
        for in_topic, out_topic, msg_type, sub_qos, pub_qos in pairs:
            pub = self.create_publisher(msg_type, out_topic, pub_qos)
            self._pubs[in_topic] = pub
            self.create_subscription(
                msg_type,
                in_topic,
                lambda msg, p=pub: self._relay(msg, p),
                sub_qos,
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
