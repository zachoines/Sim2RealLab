"""Node-level tests for the start-cell planner selector.

The cost probe is a pure function over a Costmap message and is tested
directly; the node tests cover only the gate decision and its logging.
"""

from __future__ import annotations

import array
import unittest
from unittest.mock import MagicMock

import pytest
import rclpy
from nav2_msgs.msg import BehaviorTreeLog, BehaviorTreeStatusChange, Costmap
from rclpy.parameter import Parameter

from strafer_navigation.start_cell_planner_selector import (
    INSCRIBED_INFLATED_OBSTACLE,
    NO_INFORMATION,
    StartCellPlannerSelector,
    refuses_start,
    start_cell_cost,
)


@pytest.fixture(scope="module", autouse=True)
def _rclpy_session():
    rclpy.init()
    yield
    rclpy.shutdown()


def _costmap(cells, size_x, size_y, resolution=0.05, origin=(0.0, 0.0)):
    msg = Costmap()
    msg.metadata.resolution = resolution
    msg.metadata.size_x = size_x
    msg.metadata.size_y = size_y
    msg.metadata.origin.position.x = float(origin[0])
    msg.metadata.origin.position.y = float(origin[1])
    msg.data = array.array("B", cells)
    return msg


def _make_overrides(**values):
    type_map = {
        str: Parameter.Type.STRING,
        bool: Parameter.Type.BOOL,
        int: Parameter.Type.INTEGER,
        float: Parameter.Type.DOUBLE,
    }
    return [Parameter(k, type_map[type(v)], v) for k, v in values.items()]


def _node(**overrides) -> StartCellPlannerSelector:
    return StartCellPlannerSelector(
        parameter_overrides=_make_overrides(**overrides)
    )


class TestStartCellCost(unittest.TestCase):
    """The probe reads the cell the PLANNER sees, not the one the costmap
    publishes: SmacPlanner2D downsamples, and a downsampled cell carries the
    max cost of the block it covers.
    """

    def test_reads_the_cell_under_the_robot(self) -> None:
        cm = _costmap([0, 10, 20, 30], size_x=2, size_y=2)
        self.assertEqual(start_cell_cost(cm, 0.01, 0.01), 0)
        self.assertEqual(start_cell_cost(cm, 0.06, 0.01), 10)
        self.assertEqual(start_cell_cost(cm, 0.01, 0.06), 20)
        self.assertEqual(start_cell_cost(cm, 0.06, 0.06), 30)

    def test_honours_the_costmap_origin(self) -> None:
        cm = _costmap([7, 99], size_x=2, size_y=1, origin=(-1.0, -1.0))
        self.assertEqual(start_cell_cost(cm, -0.99, -0.99), 7)
        self.assertEqual(start_cell_cost(cm, -0.94, -0.99), 99)

    def test_off_grid_is_unknowable_not_free(self) -> None:
        cm = _costmap([0, 0, 0, 0], size_x=2, size_y=2)
        self.assertIsNone(start_cell_cost(cm, 5.0, 0.0))
        self.assertIsNone(start_cell_cost(cm, -0.1, 0.0))
        self.assertIsNone(start_cell_cost(cm, 0.0, 5.0))

    def test_sub_cell_below_the_origin_is_off_grid(self) -> None:
        """Truncation toward zero would round a pose in the last cell-width
        below the origin up to cell 0 and report that cell's cost as the
        robot's. The index has to floor.
        """
        cm = _costmap([200, 0, 0, 0], size_x=2, size_y=2)   # cell (0,0) is hot
        for x in (-0.01, -0.02, -0.049):
            self.assertIsNone(start_cell_cost(cm, x, 0.01), f"x={x}")
        for y in (-0.01, -0.02, -0.049):
            self.assertIsNone(start_cell_cost(cm, 0.01, y), f"y={y}")
        self.assertEqual(start_cell_cost(cm, 0.01, 0.01), 200)  # just inside

    def test_degenerate_metadata_is_unknowable(self) -> None:
        self.assertIsNone(start_cell_cost(_costmap([], 0, 0), 0.0, 0.0))

    def test_downsample_block_takes_the_worst_cell(self) -> None:
        """A 2x2 block holding one inscribed cell is one inscribed planning
        cell, so a robot 5 cm from the halo is still refused.
        """
        cells = [0, 0, 0, INSCRIBED_INFLATED_OBSTACLE]
        cm = _costmap(cells, size_x=2, size_y=2)
        self.assertEqual(start_cell_cost(cm, 0.01, 0.01, downsampling_factor=1), 0)
        self.assertEqual(
            start_cell_cost(cm, 0.01, 0.01, downsampling_factor=2),
            INSCRIBED_INFLATED_OBSTACLE,
        )

    def test_downsample_blocks_are_origin_aligned(self) -> None:
        """Blocks tile from cell (0,0), so the block a pose falls in is
        ``mx // factor`` — not a window centred on the robot.
        """
        cells = [0, 0, 0, 0, 250, 0, 0, 0, 0]
        cm = _costmap(cells, size_x=3, size_y=3)
        self.assertEqual(start_cell_cost(cm, 0.01, 0.01, 2), 250)
        self.assertEqual(start_cell_cost(cm, 0.11, 0.11, 2), 0)

    def test_block_clipped_at_the_grid_edge(self) -> None:
        cm = _costmap([0, 0, 0, 200], size_x=2, size_y=2)
        self.assertEqual(start_cell_cost(cm, 0.06, 0.06, downsampling_factor=4), 200)


class TestRefusesStart(unittest.TestCase):
    def test_inscribed_band_and_above_refuses(self) -> None:
        self.assertTrue(refuses_start(INSCRIBED_INFLATED_OBSTACLE))
        self.assertTrue(refuses_start(254))

    def test_below_inscribed_does_not(self) -> None:
        self.assertFalse(refuses_start(0))
        self.assertFalse(refuses_start(252))

    def test_unknown_is_traversable_not_a_refusal(self) -> None:
        """The primary planner runs with allow_unknown, so an unknown cell is
        not a refusal even though it sorts above the inscribed band.
        """
        self.assertFalse(refuses_start(NO_INFORMATION))

    def test_no_reading_is_not_a_refusal(self) -> None:
        self.assertFalse(refuses_start(None))


class TestGate(unittest.TestCase):
    ARMED = dict(primary_planner_id="GridBased",
                 relaxed_planner_id="GridBasedRelaxed")

    def test_disabled_by_default_in_code(self) -> None:
        """A bare ``ros2 run`` never admits the escape hatch; the launch file
        arms it from planner_server's registered plugins.
        """
        node = _node()
        try:
            node._start_cell_cost = lambda: INSCRIBED_INFLATED_OBSTACLE
            self.assertEqual(node._relaxed, "")
            self.assertEqual(node.select(), "GridBased")
        finally:
            node.destroy_node()

    def test_admits_the_escape_hatch_only_inside_the_band(self) -> None:
        node = _node(**self.ARMED)
        try:
            node._start_cell_cost = lambda: 252
            self.assertEqual(node.select(), "GridBased")
            node._start_cell_cost = lambda: INSCRIBED_INFLATED_OBSTACLE
            self.assertEqual(node.select(), "GridBasedRelaxed")
        finally:
            node.destroy_node()

    def test_no_costmap_or_no_tf_keeps_the_primary(self) -> None:
        """Before the costmap or map->base_link arrives the probe reads
        nothing, which must select the primary rather than the escape hatch.
        """
        node = _node(**self.ARMED)
        try:
            self.assertIsNone(node._start_cell_cost())
            self.assertEqual(node.select(), "GridBased")
        finally:
            node.destroy_node()

    def test_gate_open_and_close_both_log(self) -> None:
        """Starvation stays visible: engaging warns, and so does releasing."""
        node = _node(**self.ARMED, log_throttle_s=0.0)
        try:
            node.get_logger = MagicMock(return_value=MagicMock())
            node._start_cell_cost = lambda: INSCRIBED_INFLATED_OBSTACLE
            node.select()
            self.assertEqual(node.get_logger().warning.call_count, 1)
            node._start_cell_cost = lambda: 10
            node.select()
            self.assertEqual(node.get_logger().warning.call_count, 2)
            node.select()  # already released — no repeat
            self.assertEqual(node.get_logger().warning.call_count, 2)
        finally:
            node.destroy_node()

    def test_relaxed_branch_firing_logs(self) -> None:
        node = _node(**self.ARMED, log_throttle_s=0.0)
        try:
            node.get_logger = MagicMock(return_value=MagicMock())
            node._on_bt_log(_bt_log("ComputePathToPoseRelaxed", "SUCCESS"))
            self.assertEqual(node.get_logger().warning.call_count, 1)
        finally:
            node.destroy_node()

    def test_other_bt_nodes_do_not_log(self) -> None:
        node = _node(**self.ARMED, log_throttle_s=0.0)
        try:
            node.get_logger = MagicMock(return_value=MagicMock())
            node._on_bt_log(_bt_log("ComputePathToPose", "SUCCESS"))
            node._on_bt_log(_bt_log("ComputePathToPoseRelaxed", "RUNNING"))
            node.get_logger().warning.assert_not_called()
        finally:
            node.destroy_node()

    def test_probe_publishes_the_selection(self) -> None:
        node = _node(**self.ARMED)
        try:
            node._selector_pub = MagicMock()
            node._start_cell_cost = lambda: INSCRIBED_INFLATED_OBSTACLE
            node._on_probe()
            published = node._selector_pub.publish.call_args[0][0]
            self.assertEqual(published.data, "GridBasedRelaxed")
        finally:
            node.destroy_node()


def _bt_log(node_name: str, status: str) -> BehaviorTreeLog:
    msg = BehaviorTreeLog()
    event = BehaviorTreeStatusChange()
    event.node_name = node_name
    event.previous_status = "RUNNING"
    event.current_status = status
    msg.event_log = [event]
    return msg
