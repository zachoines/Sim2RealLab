"""Run the mission generator across scenes + seeds into a mission-queue corpus.

Thin runnable wrapper over :mod:`strafer_lab.tools.build_mission_queue`. For
each scene it loads the embedded metadata + cached occupancy, generates the
per-scene rows, writes ``data/mission_queues/<scene>/queue.yaml``, and unions
them into ``data/mission_queues/corpus.yaml``. Generated rows are cached at
``data/mission_queue_cache/<scene>/<scene_seed>.json`` keyed by the generator
version + few-shot template hash + LLM seed, so a re-run under an unchanged
template is free and a changed template invalidates rather than silently
reusing.

The default run is model-free (deterministic oracle + templated paraphrases +
grounding skipped), so it is exercisable headless. Opt into the heavy passes
with ``--use-planner-llm`` / ``--use-paraphrase-llm`` / ``--ground-start-frame``;
those load their checkpoints lazily and need a GPU. The start-frame grounding
pass also needs a rendered frame, which only the Kit-launched sibling
``render_grounded_mission_corpus.py`` supplies — it boots Isaac Sim, stands up
the perception camera, and threads a live ``grounding_frame_provider`` through
:func:`run`. Run headless here with no provider and grounding is a counted
no-op: ``--ground-start-frame`` loads the VL runner but there is no frame to
ground, so every mission is skip-counted.

Every written queue is re-parsed through ``mission_queue.load_mission_queue`` as
a built-in round-trip check before the run reports success.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from strafer_lab.tools import build_mission_queue as bmq
from strafer_lab.tools.mission_queue import load_mission_queue

_SEED_RE = re.compile(r"seed(\d+)\b")


def scene_seed_from_name(scene_name: str) -> int | None:
    match = _SEED_RE.search(scene_name)
    return int(match.group(1)) if match else None


def discover_scenes(scenes_root: Path) -> list[str]:
    """Scene names with a top-level ``<scene>.usdc`` symlink under ``scenes_root``."""
    if not scenes_root.is_dir():
        return []
    return sorted(p.stem for p in scenes_root.glob("*.usdc"))


def _read_cache(cache_path: Path, header: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if cached.get("header") != header:
        return None
    return cached.get("rows")


def _write_cache(
    cache_path: Path, header: dict[str, Any], rows: list[dict[str, Any]], stats: dict[str, Any]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"header": header, "rows": rows, "stats": stats}, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def build_config(args: argparse.Namespace) -> bmq.GeneratorConfig:
    return bmq.GeneratorConfig(
        mode=args.mode,
        llm_seed=args.llm_seed,
        start_pose_seeds=args.start_pose_seeds,
        paraphrases_per_mission=args.paraphrases,
        require_groundable=args.require_groundable,
        ground_start_frame=args.ground_start_frame,
        grounding_model=args.grounding_model,
    )


def run(
    args: argparse.Namespace,
    *,
    grounding_frame_provider: Callable[[dict[str, Any], tuple[float, float, float]], Any]
    | None = None,
) -> int:
    """Generate the corpus across scenes + seeds.

    ``grounding_frame_provider`` is the seam the Kit-launched sibling injects: a
    live ``(obj, start_pose) -> frame`` callable. It is ``None`` for the headless
    path (no rendered frame), where ``--ground-start-frame`` degrades to a
    counted skip. Passed straight through to ``build_mission_queue`` alongside
    the grounding runner this function builds from ``--grounding-model``.
    """
    scenes_root = Path(args.scenes_root)
    out_root = Path(args.output_dir)
    cache_root = Path(args.cache_dir)
    config = build_config(args)
    header = bmq.cache_header(config)

    scenes = args.scenes or discover_scenes(scenes_root)
    if not scenes:
        print(f"no scenes found under {scenes_root} (pass --scenes or --scenes-root)", file=sys.stderr)
        return 2

    waypoint_runner = bmq.build_default_waypoint_runner(config.planner_model) if args.use_planner_llm else None
    paraphrase_runner = (
        bmq.build_default_paraphrase_runner(config.paraphrase_model) if args.use_paraphrase_llm else None
    )
    grounding_runner = (
        bmq.build_default_grounding_runner(config.grounding_model) if args.ground_start_frame else None
    )

    corpus: list[dict[str, Any]] = []
    all_ok = True
    for scene in scenes:
        scene_seed = scene_seed_from_name(scene)
        cache_path = cache_root / scene / f"{scene_seed}.json"
        rows = None if args.force else _read_cache(cache_path, header)
        reused = rows is not None

        if rows is None:
            try:
                inputs = bmq.load_scene_inputs(
                    scene=scene,
                    scenes_root=scenes_root,
                    scene_seed=scene_seed,
                    allow_stale_occupancy=args.allow_stale_occupancy,
                )
            except bmq.MissionGeneratorError as exc:
                print(f"[{scene}] SKIP: {exc}", file=sys.stderr)
                all_ok = all_ok and args.skip_unloadable
                continue
            result = bmq.build_mission_queue(
                inputs,
                config,
                waypoint_runner=waypoint_runner,
                paraphrase_runner=paraphrase_runner,
                grounding_runner=grounding_runner,
                grounding_frame_provider=grounding_frame_provider,
            )
            rows = result.rows
            stats = asdict(result.stats)
            _write_cache(cache_path, header, rows, stats)
            _print_scene_report(scene, result.stats, reused=False)
        else:
            _print_cached_report(scene, len(rows))

        queue_path = out_root / scene / "queue.yaml"
        bmq.write_queue(queue_path, rows)

        # Built-in round-trip check: every written row must re-parse.
        parsed = load_mission_queue(queue_path)
        if len(parsed) != len(rows):
            print(f"[{scene}] ROUND-TRIP MISMATCH: wrote {len(rows)} parsed {len(parsed)}", file=sys.stderr)
            all_ok = False
        corpus.extend(rows)

    corpus_path = out_root / "corpus.yaml"
    bmq.write_queue(corpus_path, corpus)
    parsed_corpus = load_mission_queue(corpus_path)
    print(f"\ncorpus: {len(corpus)} missions across {len(scenes)} scene(s) -> {corpus_path}")
    print(f"round-trip: {len(parsed_corpus)}/{len(corpus)} rows re-parsed via load_mission_queue")
    if len(parsed_corpus) != len(corpus):
        all_ok = False
    return 0 if all_ok else 1


def _print_scene_report(scene: str, stats: bmq.MissionGenStats, *, reused: bool) -> None:
    print(
        f"[{scene}] multi_room={stats.multi_room} emitted={stats.emitted} "
        f"(cross={stats.cross_room} same={stats.same_room}) rejected={stats.rejected} "
        f"{stats.rejected_reasons or ''} llm_retries={stats.llm_retries} "
        f"path_shape_unsatisfied={stats.path_shape_unsatisfied} "
        f"grounded(yes/no/skip)={stats.start_frame_grounded_yes}/"
        f"{stats.start_frame_grounded_no}/{stats.start_frame_grounded_skipped}"
    )


def _print_cached_report(scene: str, n: int) -> None:
    print(f"[{scene}] reused cache: {n} missions")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="*", help="explicit scene names (default: discover *.usdc)")
    p.add_argument("--scenes-root", default="Assets/generated/scenes")
    p.add_argument("--output-dir", default="data/mission_queues")
    p.add_argument("--cache-dir", default="data/mission_queue_cache")
    p.add_argument("--mode", choices=bmq.MODES, default="mixed")
    p.add_argument("--llm-seed", type=int, default=42)
    p.add_argument("--start-pose-seeds", type=int, default=1)
    p.add_argument("--paraphrases", type=int, default=3)
    p.add_argument(
        "--require-groundable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop targets whose only anchor is the un-groundable coordinate fallback "
        "(default on; --no-require-groundable emits them anyway for A/B measurement)",
    )
    p.add_argument("--use-planner-llm", action="store_true", help="load the waypoint LLM (needs a GPU)")
    p.add_argument("--use-paraphrase-llm", action="store_true", help="load the paraphrase LLM (needs a GPU)")
    p.add_argument("--ground-start-frame", action="store_true", help="run the start-frame VLM grounding pass")
    p.add_argument(
        "--grounding-model",
        default=bmq.DEFAULT_GROUNDING_MODEL,
        help="VL checkpoint for the --ground-start-frame pass. Defaults to the 7B "
        f"({bmq.DEFAULT_GROUNDING_MODEL}), which is NOT in the offline cache. Pass "
        "Qwen/Qwen2.5-VL-3B-Instruct to use the cached, offline-ready checkpoint, "
        "or pre-download the 7B first.",
    )
    p.add_argument("--allow-stale-occupancy", action="store_true", help="skip the occupancy freshness check")
    p.add_argument("--skip-unloadable", action="store_true", help="exit 0 even if some scenes failed to load")
    p.add_argument("--force", action="store_true", help="regenerate, ignoring any cached queue")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
