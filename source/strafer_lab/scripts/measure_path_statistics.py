"""Baseline the planned-path curvature and aperture-threading statistics.

CPU-only, no Kit. Measures :mod:`strafer_lab.tools.path_statistics` over two
occupancy sources so their planned paths compare like for like:

  - ``procroom``  — rooms generated on a stub env, one occupancy grid per env
  - ``infinigen`` — a scanned scene's cached occupancy sidecar + room
    footprints

The aperture threshold is not a constant. It is derived from the *body* of
the reference (scanned) clearance distribution, on the inflation-free scale,
and a sensitivity table across quantiles is printed beside it: a threshold
read off the reference's tightest doorway would read flat across every
corridor wider than that doorway, which is useless for gating a lever that
widens corridors.

Run with a python that carries numpy + torch + the package::

    <python> source/strafer_lab/scripts/measure_path_statistics.py \
        --scene <scene>.usdc --arm vanilla --arm enriched \
        --arm 'lowbias:clutter_wall_bias_prob=0.1'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from strafer_lab.tools import path_statistics as ps
from strafer_lab.tools import scene_connectivity

DEFAULT_MIN_SEPARATION_M = 1.0
DEFAULT_STRAIGHT_LINE_BAND_M = (1.5, 4.0)
RAW_REPORT_THRESHOLD_M = 0.6


# ---------------------------------------------------------------------------
# ProcRoom source
# ---------------------------------------------------------------------------


class _CaptureEntity:
    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        pass

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        pass


class _StubScene:
    def __init__(self, num_envs):
        self._entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, key):
        return self._entities[key]


def _shipped_enriched_params():
    """The generator kwargs and difficulty range the enriched variants ship."""
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    events = composed.StraferNavCfg_RLDepthEnriched_Real().events
    params = dict(events.generate_room.params)
    params.pop("collection_name", None)
    diff = events.randomize_difficulty.params
    return params, (int(diff["min_level"]), int(diff["max_level"]))


def _generate_rooms(num_envs, seed, difficulty_range, params):
    """One generator call on a stub env; returns per-env grids and spawn pools.

    The raw occupancy grid is a local inside the generator and the retry ladder
    re-rasterizes the whole batch, so it is captured by wrapping the rasterizer
    and keeping the last call — the one the stored free space derives from.
    """
    from strafer_lab.tasks.navigation.mdp import proc_room

    captured = {}
    real = proc_room._build_occupancy_grid

    def capturing(*args, **kwargs):
        out = real(*args, **kwargs)
        captured["occupancy"] = out
        return out

    lo, hi = difficulty_range
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=_StubScene(num_envs))
    torch.manual_seed(seed)
    env._proc_room_difficulty = torch.randint(lo, hi + 1, (num_envs,))
    proc_room._build_occupancy_grid = capturing
    try:
        proc_room.generate_proc_room(env, torch.arange(num_envs), **params)
    finally:
        proc_room._build_occupancy_grid = real

    origin = -proc_room.GRID_SIZE * proc_room.GRID_RES / 2.0
    return {
        "occupancy": captured["occupancy"].numpy(),
        "free_space": env._proc_room_free_space.numpy(),
        "spawn_pts": env._proc_room_spawn_pts.numpy(),
        "spawn_count": env._proc_room_spawn_count.numpy(),
        "grid_res": proc_room.GRID_RES,
        "grid_origin_xy": (origin, origin),
        "inflation_radius_m": proc_room.INFLATION_CELLS * proc_room.GRID_RES,
    }


def measure_procroom(*, vanilla, overrides, num_envs, resets, pairs_per_env,
                     seed, min_separation_m):
    if vanilla:
        params, difficulty_range = {}, (7, 7)
    else:
        params, difficulty_range = _shipped_enriched_params()
    params.update(overrides)

    rng = np.random.default_rng(seed)
    stats, failures = [], {"no_path": 0, "invalid_endpoint": 0}
    for reset_id in range(resets):
        batch = _generate_rooms(num_envs, seed + reset_id, difficulty_range, params)
        for b in range(num_envs):
            count = int(batch["spawn_count"][b])
            if count < 2:
                continue
            pairs = ps.sample_endpoint_pairs(
                batch["spawn_pts"][b][:count], rng,
                n_pairs=pairs_per_env, min_separation_m=min_separation_m,
            )
            s, f = ps.plan_over_pairs(
                pairs, batch["free_space"][b], batch["occupancy"][b],
                grid_res=batch["grid_res"],
                grid_origin_xy=batch["grid_origin_xy"],
                inflation_radius_m=batch["inflation_radius_m"],
                group=f"r{reset_id}e{b}",
            )
            stats.extend(s)
            for k in failures:
                failures[k] += f[k]
    return stats, failures, params


# ---------------------------------------------------------------------------
# Infinigen source
# ---------------------------------------------------------------------------


def measure_scene(scene_usd, *, n_pairs, seed, min_separation_m, sealed=True):
    """Measure one scanned scene.

    ``sealed`` confines both the endpoints and the corridor to the scene's
    room footprints — the region the bridge's own spawn derivation hands the
    robot. Unsealed lets a path leave the building and round its exterior,
    which is a different scene to plan in and not a mission the robot runs.
    """
    from strafer_lab.tools import scene_metadata_reader

    occupancy = scene_connectivity.load_occupancy(
        scene_connectivity.scene_dir_for(scene_usd)
    )
    rooms = scene_metadata_reader.load(scene_usd).get("rooms", [])
    if not rooms:
        raise RuntimeError(f"{scene_usd} carries no room metadata")
    free = scene_connectivity.occupancy_to_free_space(
        occupancy.grid, grid_res=occupancy.resolution_m
    )
    if sealed:
        free = scene_connectivity.seal_free_space_to_rooms(
            free, rooms, origin_xy=occupancy.origin_xy,
            grid_res=occupancy.resolution_m,
        )
        pool = np.asarray(
            scene_connectivity.spawn_pool_from_occupancy(free, rooms, occupancy),
            dtype=np.float64,
        )
    else:
        cells = np.argwhere(free).astype(np.float64)
        pool = (cells * occupancy.resolution_m
                + np.asarray(occupancy.origin_xy)
                + occupancy.resolution_m / 2.0)
    radius_cells = math.ceil(scene_connectivity.ROBOT_RADIUS_M / occupancy.resolution_m)
    pairs = ps.sample_endpoint_pairs(
        pool, np.random.default_rng(seed),
        n_pairs=n_pairs, min_separation_m=min_separation_m,
    )
    return ps.plan_over_pairs(
        pairs, free, occupancy.grid,
        grid_res=occupancy.resolution_m,
        grid_origin_xy=occupancy.origin_xy,
        inflation_radius_m=radius_cells * occupancy.resolution_m,
        group=Path(scene_usd).stem,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _parse_override(value):
    """Generator kwargs are not all numeric, so the CLI cannot assume it."""
    lowered = value.strip().lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _print_summary(label, summary, failures):
    if not summary.get("n_paths"):
        print(f"{label:>34s}  (no paths)")
        return
    td = summary["turn_density_rad_per_m"]
    to = summary["tortuosity"]
    mc = summary["min_clearance_m"]
    ec = summary.get("excess_clearance_m", {})
    print(
        f"{label:>34s}  n={summary['n_paths']:4d}/{summary['n_paths_before_band']:<5d} "
        f"turn med/p90/p99 {td['median']:.3f}/{td['p90']:.3f}/{td['p99']:.3f}  "
        f"tort med {to['median']:.3f}  "
        f"turn {summary['frac_paths_turning'] * 100:5.1f}%  "
        f"bend {summary['frac_paths_bending'] * 100:5.1f}%  "
        f"minC med {mc['median']:.3f} min {mc['min']:.3f}  "
        f"excess med {ec.get('median', float('nan')):.3f}  "
        f"fail {failures['no_path']}/{failures['invalid_endpoint']}"
        f"/{failures.get('no_obstacles', 0)}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=[],
                   metavar="LABEL[:KEY=VALUE,...]",
                   help="one procroom leg (repeatable). LABEL 'vanilla' runs the "
                        "default generator path at difficulty 7; any other label "
                        "runs the shipped enriched params with the listed kwargs "
                        "replaced")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--resets", type=int, default=1)
    p.add_argument("--pairs-per-env", type=int, default=6)
    p.add_argument("--scene", action="append", default=[],
                   help="scene USD for the reference leg (repeatable)")
    p.add_argument("--scene-pairs", type=int, default=60)
    p.add_argument("--scene-scope", choices=("sealed", "unsealed", "both"),
                   default="sealed",
                   help="confine scene paths to the room footprints (default) or "
                        "let them leave the building")
    p.add_argument("--reference", default=None,
                   help="scene USD whose clearance body sets the threshold "
                        "(default: the first --scene)")
    p.add_argument("--threshold-quantile", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--min-separation-m", type=float, default=DEFAULT_MIN_SEPARATION_M)
    p.add_argument("--straight-line-band-m", type=float, nargs=2,
                   default=DEFAULT_STRAIGHT_LINE_BAND_M,
                   help="keep only paths whose endpoints are this far apart")
    p.add_argument("--no-band", action="store_true")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    band = None if args.no_band else tuple(args.straight_line_band_m)
    legs: dict[str, tuple[list, dict]] = {}
    scene_keys: list[str] = []

    scopes = ("sealed", "unsealed") if args.scene_scope == "both" else (args.scene_scope,)
    for scene in args.scene:
        for scope in scopes:
            name = Path(scene).stem + ("" if scope == "sealed" else "/unsealed")
            if name in legs:
                p.error(f"two legs are both named {name!r}")
            legs[name] = measure_scene(
                scene, n_pairs=args.scene_pairs, seed=args.seed,
                min_separation_m=args.min_separation_m, sealed=(scope == "sealed"),
            )
            # Whichever scope was asked for is the reference; with both, the
            # sealed legs are, since that is the region production plans in.
            if scope == "sealed" or args.scene_scope == "unsealed":
                scene_keys.append(name)
            print(f"[measure] {name}: {len(legs[name][0])} paths", flush=True)

    for spec in args.arm:
        label, _, kv = spec.partition(":")
        if label in legs:
            p.error(f"two legs are both named {label!r}")
        overrides = {}
        for item in filter(None, kv.split(",")):
            if "=" not in item:
                p.error(f"--arm override {item!r} is not KEY=VALUE")
            key, _, value = item.partition("=")
            overrides[key] = _parse_override(value)
        stats, failures, params = measure_procroom(
            vanilla=(label == "vanilla"), overrides=overrides,
            num_envs=args.num_envs, resets=args.resets,
            pairs_per_env=args.pairs_per_env, seed=args.seed,
            min_separation_m=args.min_separation_m,
        )
        legs[label] = (stats, failures)
        print(f"[measure] {label}: {len(stats)} paths  params={params}", flush=True)

    if not legs:
        p.error("nothing to measure — pass at least one --scene or --arm")

    if args.reference:
        reference_keys = [Path(args.reference).stem]
        missing = [k for k in reference_keys if k not in legs]
        if missing:
            p.error(f"--reference {missing} was not measured; legs are "
                    f"{sorted(legs)}")
    else:
        reference_keys = list(scene_keys)

    excess_taus: list[float] = []
    if reference_keys:
        # The threshold is read off the same population it is applied to,
        # so the band has to be the same on both sides.
        ref = ps._band_filter(
            [s for k in reference_keys for s in legs[k][0]], band
        )
        values, weights = ps.excess_clearance(ref)
        tau = ps.threshold_from_body(ref, args.threshold_quantile)
        print(f"\nthreshold derivation — reference {'+'.join(reference_keys)}, "
              f"{len(ref)} paths, arc-weighted excess clearance (m):")
        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
            print(f"  q{int(q * 100):02d} = "
                  f"{ps.weighted_quantile(values, weights, q):.3f}"
                  + ("   <- threshold" if abs(q - args.threshold_quantile) < 1e-9 else ""))
        for k in reference_keys:
            v, w = ps.excess_clearance(ps._band_filter(legs[k][0], band))
            if len(v):
                print(f"    per-scene q50: {k} = "
                      f"{ps.weighted_quantile(v, w, 0.5):.3f}")
        print(f"  chosen tau (excess scale) = {tau:.3f} m; per-source raw threshold "
              f"= tau + that source's inflation radius")
        # tau = 0 reads the arc a robot of the source's own inflation radius
        # could not actually occupy, so it belongs beside the sweep.
        excess_taus = sorted({0.0} | {ps.weighted_quantile(values, weights, q)
                                      for q in (0.10, 0.25, 0.50, 0.75)})

    print()
    out = {}
    for label, (stats, failures) in legs.items():
        summary = ps.summarize(
            stats, excess_thresholds=excess_taus,
            raw_thresholds=(RAW_REPORT_THRESHOLD_M,), straight_line_band=band,
        )
        summary["failures"] = failures
        summary["gates"] = ps.bootstrap_gates(
            stats, excess_thresholds=excess_taus,
            raw_thresholds=(RAW_REPORT_THRESHOLD_M,), straight_line_band=band,
        )
        out[label] = summary
        _print_summary(label, summary, failures)

    print("\ngate readings with a group-resampled 95% interval:")
    for label, summary in out.items():
        g = summary.get("gates") or {}
        if not g.get("n_paths"):
            continue
        turning, bending = g["frac_paths_turning"], g["frac_paths_bending"]
        p90 = g["turn_density_p90"]

        def _ci(entry, scale=1.0, width=5, prec=1):
            if entry["ci_lo"] is None:
                return "[  no interval — one group  ]"
            return (f"[{entry['ci_lo'] * scale:{width}.{prec}f}, "
                    f"{entry['ci_hi'] * scale:{width}.{prec}f}]")

        print(f"{label:>34s}  groups={g['n_groups']:4d}  "
              f"turning {turning['value'] * 100:5.1f}% {_ci(turning, 100)}  "
              f"bending {bending['value'] * 100:5.1f}% {_ci(bending, 100)}  "
              f"turn p90 {p90['value']:.3f} {_ci(p90, 1.0, 5, 3)}")

    if excess_taus:
        print("\narc fraction below threshold (excess scale, sensitivity sweep):")
        header = "  ".join(f"tau={t:.3f}" for t in excess_taus)
        print(f"{'leg':>34s}  {header}")
        for label, summary in out.items():
            row = summary.get("arc_below_excess", {})
            cells = "  ".join(f"{row.get(ps._threshold_key(t), float('nan')) * 100:8.1f}%"
                              for t in excess_taus)
            print(f"{label:>34s}  {cells}")
        print(f"\narc fraction below the raw {RAW_REPORT_THRESHOLD_M} m threshold:")
        for label, summary in out.items():
            raw = summary.get("arc_below_raw", {})
            print(f"{label:>34s}  "
                  f"{raw.get(ps._threshold_key(RAW_REPORT_THRESHOLD_M), float('nan')) * 100:8.1f}%")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
