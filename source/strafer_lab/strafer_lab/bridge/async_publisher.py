"""Python-side ROS 2 publisher for the sim-in-the-loop non-camera telemetry.

On Isaac Lab 3.0 + Isaac Sim 6 the OmniGraph ROS 2 publishers (Clock,
Odometry, TransformTree, SubscribeTwist) must be evaluated inside Kit's
main loop, which is entered once per bridge iteration via
``simulation_app.update()``. That Kit tick also drags in the entire
Replicator / Hydra / USD pipeline and costs tens of milliseconds per
iteration, capping /clock at ~10 Hz even when env.step itself sustains
16-19 Hz.

This module replaces those specific OmniGraph nodes with a Python
``rclpy``-based publisher that runs its spinner on a dedicated thread.
Messages are serialized and dispatched to DDS inside
``publish_state(...)``, which is called from the main bridge loop after
``env.step()`` and does not depend on ``simulation_app.update()`` to
fire. The camera publishers are likewise Python-side now (see
:mod:`strafer_lab.bridge.async_camera_publisher`); the OmniGraph the
bridge builds only carries the residual ``OnPlaybackTick`` /
``ROS2Context`` scaffolding.

Only importable inside a Kit process that has activated
``isaacsim.ros2.bridge`` (which transitively loads the bundled 3.12
rclpy at ``isaacsim.ros2.core/humble/rclpy``). Importing outside Kit
will fail at the ``rclpy`` line.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import rclpy
import warp as wp
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster

from .proprio import WHEEL_JOINT_NAMES, ordered_wheel_values, resolve_wheel_indices


def _to_torch(arr):
    """Wrap an Isaac Lab 3.0 scene-data attribute in a torch tensor.

    ``Articulation.data.*`` returns ``wp.array`` on Isaac Lab 3.0 / Isaac
    Sim 6 (the old torch-tensor return was removed in the API shift).
    ``wp.to_torch`` is a zero-copy view when the array lives on CUDA, so
    this is effectively free per call.
    """
    return wp.to_torch(arr)


@dataclass
class _CmdVelState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    last_monotonic: float = 0.0


class StraferAsyncPublisher:
    """rclpy-backed publisher for /clock, /odom, odom→base TF, /cmd_vel.

    Also publishes wheel ``/strafer/joint_states`` and the D555
    ``/d555/imu/filtered`` when the caller supplies those topics (and, for
    IMU, the sensor handle) so the trained-policy obs pipeline reaches
    ``ready`` in sim-in-the-loop on the same wire contract as the robot.

    Parameters
    ----------
    robot:
        Isaac Lab ``Articulation`` handle for the Strafer chassis. Used
        to read ``root_link_pos_w`` / ``root_link_quat_w`` (world-frame
        pose, XYZW quaternion convention on Isaac Lab 3.0),
        ``root_lin_vel_b`` / ``root_ang_vel_b`` (body-frame twist), and
        ``joint_vel`` / ``joint_pos`` (wheel state) each
        ``publish_state(...)`` call.
    imu_sensor:
        Isaac Lab ``Imu`` sensor handle (the scene's ``d555_imu``), or
        ``None`` to skip IMU publishing. Read for ``lin_acc_b`` /
        ``ang_vel_b`` — the same source the training obs uses, so the
        published IMU matches the policy's training distribution.
    clock_topic, odom_topic, cmd_vel_topic:
        Topic names. Defaults match ``strafer_shared.constants`` via the
        caller — this class takes raw strings to keep imports minimal.
    joint_states_topic, imu_topic:
        Optional proprioception topics. ``None`` skips that publisher.
    odom_frame_id, base_frame_id, imu_frame_id:
        Message ``frame_id``s. ``odom_frame_id`` / ``base_frame_id`` are
        also the odom→base TF parent/child.
    node_name:
        ROS 2 node name. Defaults to ``strafer_sim_bridge_publisher``.
    """

    def __init__(
        self,
        *,
        robot,
        clock_topic: str,
        odom_topic: str,
        cmd_vel_topic: str,
        odom_frame_id: str,
        base_frame_id: str,
        imu_sensor=None,
        joint_states_topic: str | None = None,
        imu_topic: str | None = None,
        imu_frame_id: str = "d555_link",
        clock_rate_hz: float = 50.0,
        node_name: str = "strafer_sim_bridge_publisher",
    ) -> None:
        self._robot = robot
        self._odom_frame_id = odom_frame_id
        self._base_frame_id = base_frame_id
        self._imu_sensor = imu_sensor
        self._imu_frame_id = imu_frame_id
        # Resolved lazily on first publish once robot.joint_names is populated.
        self._wheel_indices: list[int] | None = None

        # rclpy may already be initialized by another part of Kit (e.g.
        # the isaacsim.ros2.core extension's startup smoke-test). Guard
        # so multiple publishers can be constructed in one process.
        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)
        self._clock_pub = self._node.create_publisher(Clock, clock_topic, 10)
        self._odom_pub = self._node.create_publisher(Odometry, odom_topic, 10)
        self._joint_states_pub = (
            self._node.create_publisher(JointState, joint_states_topic, 10)
            if joint_states_topic
            else None
        )
        # IMU needs both a topic and a sensor to read; skip otherwise.
        self._imu_pub = (
            self._node.create_publisher(Imu, imu_topic, 10)
            if (imu_topic and imu_sensor is not None)
            else None
        )
        self._tf_broadcaster = TransformBroadcaster(self._node)
        self._cmd_vel_sub = self._node.create_subscription(
            Twist, cmd_vel_topic, self._on_cmd_vel, 10
        )

        self._cmd_lock = threading.Lock()
        self._cmd = _CmdVelState(last_monotonic=time.monotonic())

        # Capture robot's starting world pose on first publish_state so
        # subsequent messages emit a pose relative to the odom origin.
        # Matches the IsaacComputeOdometry OmniGraph node's behavior,
        # which treats play-start as t=0 in the odom frame.
        self._odom_origin_xy: tuple[float, float] | None = None

        # /clock is re-published by a timer on the spinner thread at
        # ``clock_rate_hz``. The main thread updates ``_latest_sim_time_s``
        # inside publish_state after every env.step; the timer publishes
        # whatever value is current when it fires. Two timer firings
        # between main-thread updates emit the same sim_time (monotonic
        # non-decreasing, which ROS allows). Decoupling /clock from the
        # env-step loop rate restores the high-frequency /clock that
        # Kit's background main loop used to supplement on the legacy
        # OmniGraph publisher, but with deterministic rate.
        self._sim_time_lock = threading.Lock()
        self._latest_sim_time_s = 0.0
        self._have_sim_time = False

        # rclpy executor + spinner thread. Handles the cmd_vel
        # subscription and the /clock timer. publish_state still writes
        # /odom + TF directly from the main thread (those tie to
        # env.step state, so step-rate is the right cadence).
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._spin_thread = threading.Thread(
            target=self._spin, daemon=True, name="strafer_rclpy_spin"
        )
        self._spin_thread.start()

        self._clock_timer = self._node.create_timer(
            1.0 / clock_rate_hz, self._on_clock_timer
        )

    # ------------------------------------------------------------------
    # Subscription side
    # ------------------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._cmd_lock:
            self._cmd.vx = float(msg.linear.x)
            self._cmd.vy = float(msg.linear.y)
            self._cmd.wz = float(msg.angular.z)
            self._cmd.last_monotonic = time.monotonic()

    def get_cmd_vel(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``((vx, vy, 0.0), (0.0, 0.0, wz))`` — matches read_cmd_vel."""
        with self._cmd_lock:
            return ((self._cmd.vx, self._cmd.vy, 0.0), (0.0, 0.0, self._cmd.wz))

    def last_cmd_monotonic(self) -> float:
        with self._cmd_lock:
            return self._cmd.last_monotonic

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                pass

    def _on_clock_timer(self) -> None:
        """Re-publish /clock at ``clock_rate_hz`` from the spinner thread."""
        with self._sim_time_lock:
            if not self._have_sim_time:
                return
            t = self._latest_sim_time_s
        clock_msg = Clock()
        clock_msg.clock = Time(seconds=t).to_msg()
        self._clock_pub.publish(clock_msg)

    # ------------------------------------------------------------------
    # Publish side (called from main thread, once per env.step)
    # ------------------------------------------------------------------

    def publish_state(self, sim_time_s: float) -> None:
        """Publish /strafer/odom and the odom → base_link TF.

        ``sim_time_s`` is the authoritative sim clock (seconds). The
        bridge loop accumulates it via ``physics_dt * decimation`` per
        iteration to match ``IsaacReadSimulationTime``'s source. This
        method also hands the latest value to the /clock timer thread.
        """
        stamp = Time(seconds=sim_time_s).to_msg()

        with self._sim_time_lock:
            self._latest_sim_time_s = sim_time_s
            self._have_sim_time = True

        data = self._robot.data
        # data.* returns wp.array on Isaac Lab 3.0; wrap each in a torch
        # view (zero-copy on CUDA) so we can index the single-env row.
        pos = _to_torch(data.root_link_pos_w)[0]
        quat = _to_torch(data.root_link_quat_w)[0]  # XYZW on Isaac Lab 3.0
        lin = _to_torch(data.root_lin_vel_b)[0]
        ang = _to_torch(data.root_ang_vel_b)[0]

        px = float(pos[0])
        py = float(pos[1])
        pz = float(pos[2])
        qx = float(quat[0])
        qy = float(quat[1])
        qz = float(quat[2])
        qw = float(quat[3])
        vx = float(lin[0])
        vy = float(lin[1])
        vz = float(lin[2])
        wx = float(ang[0])
        wy = float(ang[1])
        wz = float(ang[2])

        if self._odom_origin_xy is None:
            self._odom_origin_xy = (px, py)
        ox = px - self._odom_origin_xy[0]
        oy = py - self._odom_origin_xy[1]

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame_id
        odom.child_frame_id = self._base_frame_id
        odom.pose.pose.position.x = ox
        odom.pose.pose.position.y = oy
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.x = wx
        odom.twist.twist.angular.y = wy
        odom.twist.twist.angular.z = wz
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._odom_frame_id
        tf.child_frame_id = self._base_frame_id
        tf.transform.translation.x = ox
        tf.transform.translation.y = oy
        tf.transform.translation.z = pz
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)

        # ── Wheel joint state ──────────────────────────────────────────
        # The inference obs pipeline reconstructs body velocity from these
        # wheel velocities via the shared mecanum inverse-FK, so publish the
        # sim articulation's wheel-joint state in canonical wheel order.
        if self._joint_states_pub is not None:
            if self._wheel_indices is None:
                self._wheel_indices = resolve_wheel_indices(self._robot.joint_names)
            jvel = _to_torch(data.joint_vel)[0]
            jpos = _to_torch(data.joint_pos)[0]
            js = JointState()
            js.header.stamp = stamp
            js.name = list(WHEEL_JOINT_NAMES)
            js.velocity = ordered_wheel_values(self._wheel_indices, jvel)
            js.position = ordered_wheel_values(self._wheel_indices, jpos)
            self._joint_states_pub.publish(js)

        # ── D555 IMU ───────────────────────────────────────────────────
        # Read the same body-frame accel/gyro the training obs uses so the
        # policy sees an in-distribution IMU. lin_acc_b includes gravity
        # (reads ~(0,0,9.81) at rest); orientation is set from the base
        # pose for completeness (the policy consumes accel/gyro only).
        if self._imu_pub is not None:
            idata = self._imu_sensor.data
            acc = _to_torch(idata.lin_acc_b)[0]
            gyr = _to_torch(idata.ang_vel_b)[0]
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = self._imu_frame_id
            imu.linear_acceleration.x = float(acc[0])
            imu.linear_acceleration.y = float(acc[1])
            imu.linear_acceleration.z = float(acc[2])
            imu.angular_velocity.x = float(gyr[0])
            imu.angular_velocity.y = float(gyr[1])
            imu.angular_velocity.z = float(gyr[2])
            imu.orientation.x = qx
            imu.orientation.y = qy
            imu.orientation.z = qz
            imu.orientation.w = qw
            self._imu_pub.publish(imu)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the spinner thread and destroy the node.

        Does NOT call ``rclpy.shutdown()`` — other Kit components may
        still hold a rclpy context. Process exit cleans up.
        """
        self._stop.set()
        try:
            self._executor.shutdown()
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
