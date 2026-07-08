#!/usr/bin/env python3
"""Observation-parity CLI for the trained-policy deployment lane.

Two modes, both joining on the sim-time axis (see scripts/PARITY_SCHEMA.md):

  --gym-dump FILE
      Join the inference node's obs-dump JSONL against a gym-side JSONL
      emitted by the DGX counterpart (the ground-truth obs the training env
      assembles per step). This is the authoritative obs-parity gate:
      scalar dims <= 1e-5, depth dims <= 1e-3.

  --self-check --bag DIR
      No DGX involvement. Re-assemble each dumped tick's obs offline from the
      bag's raw sensor topics through the SAME obs_pipeline functions the node
      uses, and join against the node dump. This pins assembly
      wiring/ordering/scales today. Because it re-samples the bag rather than
      replaying the node's exact cached inputs, non-trivial deltas on a real
      bag reflect temporal sampling, not necessarily a bug — the gym-dump join
      is the strict numerical gate. The self-check therefore defaults to looser
      advisory WIRING bounds (scalar ≤ 0.1, depth report-only); pass
      --scalar-bound / --depth-bound to override. The referent (subgoal/goal)
      and last_action are taken from the node dump itself (last_action is
      node-internal, not on any topic), so the self-check isolates the
      sensor→obs pipeline.

Exit code is 0 on PASS, 1 on FAIL (or on a usage/IO error).

Run from a sourced ROS 2 + colcon workspace so ``strafer_inference`` and the
rosbag2 / message libraries import.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

from strafer_inference import parity as P
from strafer_shared.policy_interface import PolicyVariant

_TOPIC_IMU = "/d555/imu/filtered"
_TOPIC_JOINTS = "/strafer/joint_states"
_TOPIC_ODOM = "/strafer/odom"
_TOPIC_DEPTH = "/d555/depth/image_rect_raw"


def _last_action_from_obs(obs_row: np.ndarray, variant: PolicyVariant) -> np.ndarray:
    off = 0
    for f in variant.fields:
        if f.key == "last_action":
            return np.asarray(obs_row[off : off + f.dims], dtype=np.float32)
        off += f.dims
    raise ValueError(f"variant {variant.name} has no last_action field")


def _self_check_stream(
    node_stream: P.ObsStream,
    bag: dict[str, list],
    *,
    map_frame: str,
    base_frame: str,
) -> tuple[P.ObsStream, int]:
    """Re-assemble a reference obs stream from bag raw topics at each node
    tick's sim time. Returns (reference stream, ticks dropped for missing input)."""
    from strafer_inference import bag_io

    variant = node_stream.variant
    has_depth = any(f.key == "depth_image" for f in variant.fields)

    imu_s, imu_m = bag_io.by_stamp(bag.get(_TOPIC_IMU, []))
    jnt_s, jnt_m = bag_io.by_stamp(bag.get(_TOPIC_JOINTS, []))
    odo_s, odo_m = bag_io.by_stamp(bag.get(_TOPIC_ODOM, []))
    dep_s, dep_m = bag_io.by_stamp(bag.get(_TOPIC_DEPTH, [])) if has_depth else ([], [])
    tf_buf = bag_io.build_tf_buffer(bag.get("/tf", []), bag.get("/tf_static", []))

    recs: list[dict] = []
    dropped = 0
    for i in range(len(node_stream)):
        t = float(node_stream.t_sim[i])
        imu = bag_io.latest_at(imu_s, imu_m, t)
        jnt = bag_io.latest_at(jnt_s, jnt_m, t)
        odo = bag_io.latest_at(odo_s, odo_m, t)
        ref = node_stream.referent_xy[i]
        base = bag_io.lookup_base_in_map(tf_buf, map_frame, base_frame, t)
        depth = bag_io.latest_at(dep_s, dep_m, t) if has_depth else None
        # Mirror the node's _on_depth: a non-32FC1 depth frame is not usable.
        bad_depth = has_depth and depth is not None and depth.encoding != "32FC1"
        if (
            imu is None
            or jnt is None
            or odo is None
            or base is None
            or np.isnan(ref).any()
            or (has_depth and depth is None)
            or bad_depth
        ):
            dropped += 1
            continue

        base_xy, base_quat = base
        try:
            depth_meters = None
            if has_depth:
                depth_meters = np.frombuffer(depth.data, dtype=np.float32).reshape(
                    depth.height, depth.width
                )
            obs = P.reassemble_obs_from_extracted(
                variant,
                imu_accel=(
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ),
                imu_gyro=(
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ),
                joint_names=list(jnt.name),
                joint_velocities=list(jnt.velocity),
                body_velocity_xy=(odo.twist.twist.linear.x, odo.twist.twist.linear.y),
                last_action=_last_action_from_obs(node_stream.obs[i], variant),
                referent_map_xy=(float(ref[0]), float(ref[1])),
                base_in_map_xy=base_xy,
                base_in_map_quat=base_quat,
                depth_meters=depth_meters,
            )
        except (ValueError, KeyError):
            # Off-resolution/truncated depth or an unparseable joint state drops
            # the tick — matching the node, which skips rather than aborting.
            dropped += 1
            continue
        recs.append({"t_sim": t, "variant": variant.name, "obs": obs.tolist()})

    if not recs:
        raise SystemExit(
            "self-check reconstructed 0 ticks — the bag is missing a required "
            "topic (IMU / joint_states / odom / TF"
            + (" / depth" if has_depth else "")
            + ") over the dumped window."
        )
    return P.parse_obs_records(recs, source="self-check"), dropped


def _report(node_stream: P.ObsStream, ref_stream: P.ObsStream, args) -> bool:
    obs_report = P.compute_obs_parity(
        node_stream,
        ref_stream,
        scalar_bound=args.scalar_bound,
        depth_bound=args.depth_bound,
        min_matched_fraction=args.min_matched_fraction,
    )
    print(P.format_obs_report(obs_report))

    if obs_report.depth_max_abs is not None and obs_report.join.n_matched > 0:
        depth_report = P.depth_spatial_residual(
            node_stream, ref_stream, obs_report.join
        )
        if depth_report is not None:
            print(P.format_depth_report(depth_report))

    print(P.format_cadence_report(P.cadence_report(node_stream.t_sim)))
    return obs_report.passed


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--node-dump", required=True, help="inference node obs-dump JSONL")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gym-dump", help="gym-side obs JSONL to join against")
    src.add_argument(
        "--self-check",
        action="store_true",
        help="re-assemble the reference from --bag raw topics",
    )
    ap.add_argument("--bag", help="rosbag2 directory (required with --self-check)")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_link")
    # Default None so the mode picks the bound: strict for --gym-dump, looser
    # wiring tolerances for --self-check (which re-samples the bag).
    ap.add_argument("--scalar-bound", type=float, default=None)
    ap.add_argument("--depth-bound", type=float, default=None)
    ap.add_argument(
        "--min-matched-fraction", type=float, default=P.MIN_MATCHED_FRACTION
    )
    args = ap.parse_args(argv)

    if args.scalar_bound is None:
        args.scalar_bound = (
            P.SELF_CHECK_SCALAR_BOUND if args.self_check else P.OBS_SCALAR_BOUND
        )
    if args.depth_bound is None:
        args.depth_bound = (
            P.SELF_CHECK_DEPTH_BOUND if args.self_check else P.OBS_DEPTH_BOUND
        )

    node_stream = P.load_obs_jsonl(args.node_dump, source="node-dump")
    jumps = P.nonmonotonic_jumps(node_stream.t_sim)
    if jumps:
        print(
            f"WARNING: node dump t_sim goes backward {jumps} time(s) — this "
            "looks like two runs concatenated into one file (sim clock reset). "
            "The parity verdict below may be contaminated; re-capture one run."
        )

    if args.gym_dump:
        ref_stream = P.load_obs_jsonl(args.gym_dump, source="gym-dump")
    else:
        if not args.bag:
            ap.error("--self-check requires --bag")
        print(
            "SELF-CHECK MODE: bounds are advisory WIRING tolerances "
            f"(scalar ≤ {args.scalar_bound:g}, depth ≤ {args.depth_bound:g}) — "
            "this re-samples the bag rather than the node's exact cached inputs, "
            "so it catches gross wiring/ordering/scale bugs, not float-noise "
            "parity. The --gym-dump join is the strict numeric gate."
        )
        from strafer_inference import bag_io

        bag = bag_io.read_bag(
            args.bag,
            {_TOPIC_IMU, _TOPIC_JOINTS, _TOPIC_ODOM, _TOPIC_DEPTH, "/tf", "/tf_static"},
        )
        ref_stream, dropped = _self_check_stream(
            node_stream, bag, map_frame=args.map_frame, base_frame=args.base_frame
        )
        if dropped:
            print(
                f"note: {dropped} dumped tick(s) had no reconstructable bag "
                "input and were skipped from the reference."
            )

    passed = _report(node_stream, ref_stream, args)
    print("=" * 60)
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
