"""Name Nav2's planner from the cost of the robot's own global-costmap cell.

``SmacPlanner2D`` refuses a start cell at or above the inscribed value, and no
recovery in the navigate-to-pose tree can translate a robot out of an inflation
halo, so the mission wedges. This node answers the one question the behavior
tree has no vocabulary to ask -- is the robot's own cell in the halo? -- and
names the escape-hatch planner on ``planner_selector`` while it is, so the tree
admits that planner for this cause alone.

The probe reproduces the planner's own view rather than the raw costmap:
SmacPlanner2D plans on a downsampled grid whose cells carry the MAX cost of the
block they cover, so a single-cell probe under-reports exactly the poses the
escape hatch exists for.

An empty ``relaxed_planner_id`` disables the node: it then publishes the
primary planner unconditionally, which is what the behavior tree's
``default_planner`` already assumes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
import tf2_ros
from nav2_msgs.msg import BehaviorTreeLog, Costmap
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# nav2_costmap_2d/cost_values.hpp. INSCRIBED is the band SmacPlanner2D's
# collision checker rejects; NO_INFORMATION is traversable under the primary
# planner's allow_unknown, so it is not a refusal even though it sorts highest.
INSCRIBED_INFLATED_OBSTACLE = 253
NO_INFORMATION = 255


def start_cell_cost(
    costmap: Costmap, x: float, y: float, downsampling_factor: int = 1
) -> Optional[int]:
    """Cost of the planning cell containing ``(x, y)``, or None if off-grid.

    The max over the downsample block, matching ``CostmapDownsampler``: the
    planner sees one cell where the costmap has ``factor ** 2``.
    """
    meta = costmap.metadata
    if meta.resolution <= 0.0 or meta.size_x == 0 or meta.size_y == 0:
        return None
    # floor, not int(): int() truncates toward zero, so a pose within one cell
    # below the origin would land on cell 0 instead of reading off-grid.
    mx = math.floor((x - meta.origin.position.x) / meta.resolution)
    my = math.floor((y - meta.origin.position.y) / meta.resolution)
    if not (0 <= mx < meta.size_x and 0 <= my < meta.size_y):
        return None

    factor = max(1, int(downsampling_factor))
    x0 = (mx // factor) * factor
    y0 = (my // factor) * factor
    worst = 0
    for j in range(y0, min(y0 + factor, meta.size_y)):
        row = j * meta.size_x
        for i in range(x0, min(x0 + factor, meta.size_x)):
            worst = max(worst, costmap.data[row + i])
    return worst


def refuses_start(cost: Optional[int]) -> bool:
    """Whether SmacPlanner2D would reject a start cell of this cost."""
    if cost is None or cost == NO_INFORMATION:
        return False
    return cost >= INSCRIBED_INFLATED_OBSTACLE


class StartCellPlannerSelector(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("start_cell_planner_selector", **kwargs)

        self.declare_parameter("primary_planner_id", "GridBased")
        self.declare_parameter("relaxed_planner_id", "")
        self.declare_parameter("costmap_topic", "global_costmap/costmap_raw")
        self.declare_parameter("planner_selector_topic", "planner_selector")
        self.declare_parameter("behavior_tree_log_topic", "behavior_tree_log")
        self.declare_parameter("relaxed_bt_node_name", "ComputePathToPoseRelaxed")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("probe_period_s", 0.5)
        self.declare_parameter("downsampling_factor", 1)
        self.declare_parameter("log_throttle_s", 10.0)

        self._primary = str(self.get_parameter("primary_planner_id").value)
        self._relaxed = str(self.get_parameter("relaxed_planner_id").value)
        self._global_frame = str(self.get_parameter("global_frame").value)
        self._base_frame = str(self.get_parameter("robot_base_frame").value)
        self._factor = int(self.get_parameter("downsampling_factor").value)
        self._throttle_s = float(self.get_parameter("log_throttle_s").value)
        self._relaxed_bt_node = str(
            self.get_parameter("relaxed_bt_node_name").value
        )

        self._costmap: Optional[Costmap] = None
        self._admitting = False

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # PlannerSelector caches the last message it saw, so a durable
        # publisher plus a steady republish converges the tree whether its
        # subscription is durable or joins late.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._selector_pub = self.create_publisher(
            String,
            str(self.get_parameter("planner_selector_topic").value),
            latched,
        )
        self.create_subscription(
            Costmap, str(self.get_parameter("costmap_topic").value),
            self._on_costmap, 1,
        )
        self.create_subscription(
            BehaviorTreeLog,
            str(self.get_parameter("behavior_tree_log_topic").value),
            self._on_bt_log, 10,
        )
        self.create_timer(
            float(self.get_parameter("probe_period_s").value), self._on_probe
        )

        self.get_logger().info(
            f"start-cell planner selector: primary={self._primary!r} "
            f"relaxed={self._relaxed or '<disabled>'!r} "
            f"downsampling_factor={self._factor}"
        )

    def _on_costmap(self, msg: Costmap) -> None:
        self._costmap = msg

    def _lookup_robot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._global_frame, self._base_frame, rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return None
        return (
            float(tf.transform.translation.x),
            float(tf.transform.translation.y),
        )

    def _start_cell_cost(self) -> Optional[int]:
        if self._costmap is None:
            return None
        xy = self._lookup_robot_xy()
        if xy is None:
            return None
        return start_cell_cost(self._costmap, xy[0], xy[1], self._factor)

    def select(self) -> str:
        """Planner the tree should use for its next attempt."""
        if not self._relaxed:
            return self._primary
        cost = self._start_cell_cost()
        admit = refuses_start(cost)
        self._note_gate(admit, cost)
        return self._relaxed if admit else self._primary

    def _note_gate(self, admit: bool, cost: Optional[int]) -> None:
        if admit:
            self.get_logger().warning(
                f"Robot's own global-costmap cell is {cost} (>= inscribed "
                f"{INSCRIBED_INFLATED_OBSTACLE}); {self._primary!r} cannot "
                f"plan from it. Admitting {self._relaxed!r} until the robot "
                "is clear.",
                throttle_duration_sec=self._throttle_s,
            )
        elif self._admitting:
            self.get_logger().warning(
                f"Robot's own global-costmap cell is {cost}, back under the "
                f"inscribed band; {self._primary!r} only."
            )
        self._admitting = admit

    def _on_probe(self) -> None:
        msg = String()
        msg.data = self.select()
        self._selector_pub.publish(msg)

    def _on_bt_log(self, msg: BehaviorTreeLog) -> None:
        for event in msg.event_log:
            if (
                event.node_name == self._relaxed_bt_node
                and event.current_status == "SUCCESS"
            ):
                self.get_logger().warning(
                    f"{self._relaxed!r} produced this goal's path: the nav2 "
                    "lane is planning from a start the primary planner "
                    "refuses.",
                    throttle_duration_sec=self._throttle_s,
                )
                return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StartCellPlannerSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
