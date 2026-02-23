"""Depth downsampler node for policy input.

Subscribes to full-resolution depth from the RealSense D555 camera,
resizes to the policy input resolution (80x60), clips to the valid
depth range, and publishes the result for the inference node.

Input:  /d555/depth/image_rect_raw  (16UC1, millimeters, native resolution)
Output: /d555/depth/downsampled     (32FC1, meters, 80x60)
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from strafer_shared.constants import (
    DEPTH_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_CLIP_NEAR,
    DEPTH_CLIP_FAR,
)


def process_depth(raw_16uc1: np.ndarray) -> np.ndarray:
    """Convert raw 16UC1 depth (mm) to clipped, downsampled float32 (m).

    Args:
        raw_16uc1: uint16 array in millimeters (RealSense depth output).

    Returns:
        float32 array of shape (DEPTH_HEIGHT, DEPTH_WIDTH) in meters,
        with out-of-range values zeroed.
    """
    depth_m = raw_16uc1.astype(np.float32) * 0.001
    depth_m[(depth_m < DEPTH_CLIP_NEAR) | (depth_m > DEPTH_CLIP_FAR)] = 0.0
    return cv2.resize(
        depth_m,
        (DEPTH_WIDTH, DEPTH_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


class DepthDownsampler(Node):

    def __init__(self):
        super().__init__("depth_downsampler")

        self._bridge = CvBridge()
        self._frame_count = 0

        self._sub = self.create_subscription(
            Image,
            "/d555/depth/image_rect_raw",
            self._depth_callback,
            10,
        )

        self._pub = self.create_publisher(
            Image,
            "/d555/depth/downsampled",
            10,
        )

        self.get_logger().info(
            f"Depth downsampler ready: "
            f"{DEPTH_WIDTH}x{DEPTH_HEIGHT}, "
            f"clip [{DEPTH_CLIP_NEAR}, {DEPTH_CLIP_FAR}]m"
        )

    def _depth_callback(self, msg: Image):
        # RealSense publishes depth as 16UC1 (millimeters)
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_resized = process_depth(cv_image)

        # Publish as 32FC1
        out_msg = self._bridge.cv2_to_imgmsg(depth_resized, encoding="32FC1")
        out_msg.header = msg.header
        self._pub.publish(out_msg)

        self._frame_count += 1
        if self._frame_count == 1:
            self.get_logger().info(
                f"First depth frame received: {msg.width}x{msg.height} → "
                f"{DEPTH_WIDTH}x{DEPTH_HEIGHT}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DepthDownsampler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
