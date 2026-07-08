"""rosbag2 + TF reading glue for the parity CLIs.

Kept OUT of the pure ``parity`` library (and off the unit-test import path):
importing this module pulls rosbag2_py / tf2_ros / rclpy, which exist only in a
sourced ROS 2 environment. The parity math stays rclpy-free in ``parity.py``;
this module only turns a bag on disk into the plain numbers that library
consumes.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Optional


def read_bag(bag_uri: str, topics: set[str]) -> dict[str, list]:
    """Read the requested topics from a rosbag2 directory into per-topic
    message lists, in record order."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_uri, storage_id=""),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out: dict[str, list] = {t: [] for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in topics:
            out[topic].append(deserialize_message(data, get_message(type_map[topic])))
    return out


def header_stamp_s(msg) -> float:
    """Sim-time seconds from a message's ``header.stamp`` (the parity axis)."""
    st = msg.header.stamp
    return st.sec + st.nanosec * 1e-9


def by_stamp(msgs: list) -> tuple[list[float], list]:
    """Sort messages by header stamp; return parallel ``(stamps, msgs)``."""
    pairs = sorted(((header_stamp_s(m), m) for m in msgs), key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def latest_at(stamps: list[float], msgs: list, t: float):
    """Message with the greatest header stamp ``<= t`` (the value a
    latest-wins cache held at that sim instant), or ``None`` if all are ``> t``."""
    idx = bisect_right(stamps, t) - 1
    if idx < 0:
        return None
    return msgs[idx]


def build_tf_buffer(tf_msgs: list, tf_static_msgs: list, cache_s: float = 10_000.0):
    """A tf2 buffer preloaded from recorded ``/tf`` + ``/tf_static``.

    The cache is set far longer than any parity bag so a lookup at any tick
    stamp resolves against the loaded history."""
    import tf2_ros
    from rclpy.duration import Duration

    buf = tf2_ros.Buffer(cache_time=Duration(seconds=cache_s))
    for msg in tf_static_msgs:
        for tr in msg.transforms:
            buf.set_transform_static(tr, "bag")
    for msg in tf_msgs:
        for tr in msg.transforms:
            buf.set_transform(tr, "bag")
    return buf


def lookup_base_in_map(
    buf, map_frame: str, base_frame: str, t: float
) -> Optional[tuple[tuple[float, float], tuple[float, float, float, float]]]:
    """``(base_xy, base_quat_xyzw)`` for ``map->base_frame`` at sim time ``t``.

    Tries the exact stamp first (an interpolated TF); on extrapolation falls
    back to the latest available transform, matching the nodes'
    ``lookup_transform(..., Time())`` latest-wins behaviour. ``None`` if the
    transform is unavailable at either query."""
    import rclpy.time
    import tf2_ros

    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    queries = (
        rclpy.time.Time(seconds=sec, nanoseconds=nanosec),
        rclpy.time.Time(),
    )
    for query in queries:
        try:
            tf = buf.lookup_transform(map_frame, base_frame, query)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            continue
        tr = tf.transform.translation
        rot = tf.transform.rotation
        return (tr.x, tr.y), (rot.x, rot.y, rot.z, rot.w)
    return None
