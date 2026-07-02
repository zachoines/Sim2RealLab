"""Node-level tests for the subgoal-generator's /plan-staleness guard.

Spins up rclpy long enough to construct the node; the rolling-subgoal
selection math itself is covered rclpy-free in test_generator.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from strafer_inference.subgoal_generator_node import SubgoalGeneratorNode


@pytest.fixture(scope="module", autouse=True)
def _rclpy_session():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_overrides(**values) -> list[Parameter]:
    type_map = {
        str: Parameter.Type.STRING,
        bool: Parameter.Type.BOOL,
        int: Parameter.Type.INTEGER,
        float: Parameter.Type.DOUBLE,
    }
    return [Parameter(k, type_map[type(v)], v) for k, v in values.items()]


def _node(**overrides) -> SubgoalGeneratorNode:
    return SubgoalGeneratorNode(
        parameter_overrides=_make_overrides(path_timeout_s=1.0, **overrides)
    )


class TestPlanFresh(unittest.TestCase):
    def test_none_is_stale(self) -> None:
        node = _node()
        try:
            self.assertFalse(node._plan_fresh(100.0))  # never received a plan
        finally:
            node.destroy_node()

    def test_within_timeout_is_fresh(self) -> None:
        node = _node()
        try:
            node._last_plan_rx_t = 100.0
            self.assertTrue(node._plan_fresh(100.5))     # 0.5 s <= 1.0 s budget
            self.assertTrue(node._plan_fresh(101.0))     # 1.0 s == budget
            self.assertFalse(node._plan_fresh(101.001))  # just past budget
        finally:
            node.destroy_node()


class TestStalePlanSuppressesSubgoal(unittest.TestCase):
    def test_tick_suppresses_publish_when_plan_stale(self) -> None:
        node = _node()
        try:
            # has_path True (a plan was installed) but its receipt is ancient.
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._last_plan_rx_t = 0.0  # far older than path_timeout_s
            pub = MagicMock()
            node._subgoal_pub = pub
            node._on_tick()
            pub.publish.assert_not_called()
            self.assertTrue(node._stale_plan_logged)
        finally:
            node.destroy_node()

    def test_no_path_returns_before_staleness_check(self) -> None:
        node = _node()
        try:
            pub = MagicMock()
            node._subgoal_pub = pub
            node._on_tick()  # no plan yet -> early return, no publish, no crash
            pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_fresh_tick_resets_stale_log_flag(self) -> None:
        import time

        node = _node()
        try:
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._subgoal_pub = MagicMock()
            node._last_plan_rx_t = 0.0  # stale
            node._on_tick()
            self.assertTrue(node._stale_plan_logged)
            # A fresh plan resets the flag (so the warning can re-fire on a
            # future stale transition); the reset happens before the TF lookup.
            node._last_plan_rx_t = time.monotonic()
            node._on_tick()  # TF lookup fails (no transform) -> returns
            self.assertFalse(node._stale_plan_logged)
        finally:
            node.destroy_node()

    def test_tf_not_consulted_while_plan_stale(self) -> None:
        node = _node()
        try:
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._subgoal_pub = MagicMock()
            node._tf_buffer = MagicMock()
            node._last_plan_rx_t = 0.0  # stale -> suppress before TF
            node._on_tick()
            node._tf_buffer.lookup_transform.assert_not_called()
        finally:
            node.destroy_node()


class TestComposedStaleBound(unittest.TestCase):
    """Regression anchor for the stale-plan -> STOP bound. The two serial
    stages each carry ~half of the trust budget so they compose to ~2.0 s:
    the generator stops publishing the subgoal one stage after /plan dies,
    then the inference watchdog trips one stage later. If either default
    drifts back to the full budget the composed bound doubles, so pin the
    sum here.
    """

    def test_generator_and_inference_budgets_compose_to_two_seconds(self) -> None:
        from strafer_inference.watchdog import WatchdogTimeouts

        # Build WITHOUT a path_timeout_s override so _path_timeout_s reflects
        # the DECLARED default; this fails if the generator's param default
        # drifts back to the full budget (which would double the composed
        # bound). The _node() helper injects 1.0, which would mask that.
        node = SubgoalGeneratorNode(parameter_overrides=[])
        try:
            generator_budget = node._path_timeout_s
        finally:
            node.destroy_node()
        # Inference half: the WatchdogTimeouts.path default the inference node
        # feeds from its own path_timeout_s param.
        inference_budget = WatchdogTimeouts(
            imu=0.2, joint_states=0.2, odom=0.2, depth=0.5, tf=0.5,
        ).path
        self.assertEqual(generator_budget, pytest.approx(1.0))
        self.assertEqual(inference_budget, pytest.approx(1.0))
        self.assertEqual(
            generator_budget + inference_budget, pytest.approx(2.0)
        )
