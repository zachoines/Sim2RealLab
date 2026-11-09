"""``JetsonRosClient.close`` stops the spin thread before it shuts the executor down."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from strafer_autonomy.clients.ros_client import JetsonRosClient, RosClientConfig


class _ExitRequested(Exception):
    pass


class _FakeExecutor:
    def __init__(self, client: JetsonRosClient, *, block: threading.Event | None = None) -> None:
        self.client = client
        self.block = block
        self.spins = 0
        self.raise_on_spin: BaseException | None = None
        self.calls: list[tuple[str, bool]] = []

    def spin_once(self, timeout_sec: float) -> None:
        self.spins += 1
        if self.raise_on_spin is not None:
            raise self.raise_on_spin
        if self.block is not None:
            self.block.wait()
        else:
            threading.Event().wait(timeout_sec)

    def shutdown(self) -> None:
        self.calls.append(("shutdown", self.client._spin_thread.is_alive()))


def _client(*, block: threading.Event | None = None, join_timeout_s: float = 1.0, start: bool = True):
    client = JetsonRosClient.__new__(JetsonRosClient)
    executor = _FakeExecutor(client, block=block)
    client._config = RosClientConfig(spin_wake_period_s=0.01, spin_join_timeout_s=join_timeout_s)
    client._executor = executor
    client._node = MagicMock()
    client._spin_exits_on = (_ExitRequested,)
    if start:
        client._start_spin_thread()
    return client, executor


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestClose(unittest.TestCase):

    def setUp(self) -> None:
        self.thread_errors: list[BaseException] = []
        hook = patch("threading.excepthook", side_effect=lambda args: self.thread_errors.append(args.exc_value))
        hook.start()
        self.addCleanup(hook.stop)

    def test_the_thread_ends_by_returning_before_the_executor_shuts_down(self) -> None:
        client, executor = _client()
        self.assertTrue(_wait_until(lambda: executor.spins >= 2))
        self.assertTrue(client._spin_thread.is_alive())
        client.close()
        self.assertFalse(client._spin_thread.is_alive())
        self.assertEqual(self.thread_errors, [])
        self.assertEqual(executor.calls, [("shutdown", False)])
        client._node.destroy_node.assert_called_once_with()

    def test_a_second_close_does_nothing(self) -> None:
        client, executor = _client()
        self.assertTrue(_wait_until(lambda: executor.spins >= 1))
        client.close()
        client.close()
        self.assertEqual(executor.calls, [("shutdown", False)])
        client._node.destroy_node.assert_called_once_with()

    def test_a_thread_that_does_not_stop_leaves_the_executor_and_node_alone(self) -> None:
        block = threading.Event()
        client, executor = _client(block=block, join_timeout_s=0.05)
        try:
            self.assertTrue(_wait_until(lambda: executor.spins >= 1))
            with self.assertLogs("strafer_autonomy.clients.ros_client", level="WARNING"):
                client.close()
            self.assertTrue(client._spin_thread.is_alive())
            self.assertEqual(executor.calls, [])
            client._node.destroy_node.assert_not_called()
        finally:
            block.set()
            client._spin_thread.join(1.0)
        self.assertEqual(self.thread_errors, [])

    def test_an_external_shutdown_ends_the_thread_by_returning(self) -> None:
        client, executor = _client()
        self.assertTrue(_wait_until(lambda: executor.spins >= 1))
        executor.raise_on_spin = _ExitRequested()
        client._spin_thread.join(1.0)
        self.assertFalse(client._spin_thread.is_alive())
        self.assertEqual(self.thread_errors, [])
        client.close()
        self.assertEqual(executor.calls, [("shutdown", False)])

    def test_a_start_interrupted_after_the_thread_starts_closes_it(self) -> None:
        client, executor = _client(start=False)
        real_start = JetsonRosClient._start_spin_thread

        def start_then_interrupt() -> None:
            real_start(client)
            raise KeyboardInterrupt

        with patch.object(client, "_start_spin_thread", side_effect=start_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                client._start()
        self.assertFalse(client._spin_thread.is_alive())
        self.assertEqual(self.thread_errors, [])
        self.assertEqual(executor.calls, [("shutdown", False)])
        client._node.destroy_node.assert_called_once_with()

    def test_a_thread_that_never_started_closes_cleanly(self) -> None:
        client, executor = _client(start=False)
        client._spin_stop = threading.Event()
        client._closed = False
        client._spin_thread = threading.Thread(target=client._spin, daemon=True)
        client.close()
        self.assertEqual(executor.calls, [("shutdown", False)])
        client._node.destroy_node.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
