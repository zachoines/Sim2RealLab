"""Kit-launched sibling of ``build_mission_corpus.py`` that grounds start frames.

The headless ``build_mission_corpus.py`` cannot ground: the start-frame
visibility check needs a rendered frame's Replicator annotators, and
``start_pose`` is derived mid-traversal off a single stateful RNG, so the poses
cannot be enumerated and pre-rendered to disk without re-running the generator
(any drift becomes a silent skip). This script closes that gap by rendering the
frame *in the same traversal that computes the pose*: it boots Isaac Sim once
via ``AppLauncher``, stands up the perception-camera env around one scene,
builds a live ``grounding_frame_provider`` (see
:func:`strafer_lab.tools.grounding_frame_provider.make_render_grounding_frame_provider`),
and hands it to :func:`build_mission_corpus.run`, which threads it — alongside
the model-free geometric grounding runner — into ``build_mission_queue``. The
grounding verdict is GEOMETRIC: at generation time the target instance and scene
geometry are known ground truth, so visibility is read off the
``bounding_box_2d_tight`` (occlusion + size) and ``instance_id_segmentation``
(per-instance identity, joined by the target's USD prim path) annotators — NO VL
model co-resident with Kit. A re-rolled seed (a same-room "no") renders natively
when the loop reaches the next seeded pose; there is no pose key to miss.

It reuses ``build_mission_corpus``'s parser and ``run`` verbatim, so every
corpus flag (``--mode`` / ``--llm-seed`` / ``--start-pose-seeds`` /
``--output-dir`` / ``--cache-dir`` / ``--force`` / ``--require-groundable``
...) behaves identically; this script only adds the Kit boot, the live frame
source, and a few render knobs.

One scene per Kit boot. The env is built around a single scene's geometry —
rendering one scene's poses against another's geometry would be wrong — so this
grounds exactly one scene. For a multi-scene corpus, run it once per scene (the
same per-scene-boot pattern the bridge/teleop capture drivers use); each writes
its scene's ``queue.yaml``. Boot is once *per run*, never per mission.

Operator runbook (this is a Kit step — the agent never launches it). The gate
needs NO VL model, NO ``--grounding-model``, and NO env mutation (the geometric
verdict has no torch-VL dependency, so the cuda-bindings conflict is gone):

    source env_setup.sh
    isaaclab -p source/strafer_lab/scripts/render_grounded_mission_corpus.py \\
        --scenes <one_scene> \\
        --ground-start-frame \\
        --start-pose-seeds 5 \\
        --force \\
        --headless

  - ``--ground-start-frame`` is REQUIRED here — without it there is no runner to
    build and grounding is skipped, which defeats the script's only purpose.
  - ``--start-pose-seeds`` defaults to 1, which gives a same-room "no" no
    alternate seed and silently drops that target. Pass > 1 (5 here) so a "no"
    re-rolls to the next seeded start pose instead of dropping the mission.
  - ``--force`` (or a fresh ``--cache-dir``) is mandatory: the corpus cache key
    omits ``ground_start_frame``, so a grounded run after any prior ungrounded
    run reuses the stale ungrounded rows, the provider never fires, and
    grounding stays all-skipped.
  - Report the per-scene ``start_frame_grounded`` yes/no/skip rate (the
    per-scene line ``grounded(yes/no/skip)=...``). That rate — now a
    deterministic geometric visibility count, not a model judgement — is the
    measurement that closes the start-frame half of the grounding gate.
  - If the teleported robot has not settled before the annotator read, raise
    ``--grounding-warmup-steps``; for Infinigen floors above world z=0, raise
    ``--spawn-z`` to ``floor_top_z + wheel_clearance``.

  The ``instance_id_segmentation`` schema is verified against the installed
  Isaac Sim (``(H, W, 1)`` uint32 id mask + ``info["idToLabels"]`` mapping id ->
  mesh prim path, ``colorize=False``). On the first live run, sanity-check the
  yes/no/skip distribution against a few rendered frames; if Infinigen objects
  map to mesh prims this code does not resolve, adjust
  ``segment_ids_for_prim_path`` in ``tools/bbox_extractor.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ``build_mission_corpus`` is the sibling module in this scripts/ dir. Its import
# pulls the (Kit-free) generator + task registry — the same chain the headless
# corpus run and the harness suite exercise without Isaac Sim — so it is safe
# before the AppLauncher boot, and it registers the gym task at import time. All
# actual sim construction (parse_env_cfg / gym.make) stays strictly post-boot.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Default Isaac Lab task carrying the 640x360 ``d555_camera_perception`` camera
# (composed via ``StraferNavCfg_BridgeAutonomy``). num_envs is forced to 1.
DEFAULT_TASK = "Isaac-Strafer-Nav-Capture-Bridge-v0"


def _resolve_single_scene(args) -> str:
    """Return the one scene to ground, or exit with guidance.

    The env binds to one scene's geometry, so grounding is single-scene. Require
    the operator to name exactly one scene rather than auto-discovering several
    and silently grounding only the first.
    """
    scenes = list(args.scenes or [])
    if len(scenes) == 1:
        return scenes[0]
    raise SystemExit(
        "render_grounded_mission_corpus.py grounds one scene per Kit boot; pass "
        "exactly one scene via --scenes <scene>. For a multi-scene corpus, run "
        f"this once per scene. (got --scenes {scenes or 'unset'})"
    )


def main() -> int:
    import build_mission_corpus
    from strafer_lab.tools import grounding_frame_provider as gfp

    # Reuse the corpus parser verbatim so no corpus flag drifts, then add the
    # Kit boot + render knobs this driver needs.
    parser = build_mission_corpus.build_parser()
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Isaac Lab task carrying the 640x360 perception camera; num_envs "
        "forced to 1 for the render.",
    )
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=gfp.DEFAULT_SPAWN_Z,
        help="Robot root z (m above env origin) when teleported for the render. "
        "Default assumes the floor sits at world z=0; raise to "
        "floor_top_z + wheel clearance for Infinigen floors above origin.",
    )
    parser.add_argument(
        "--grounding-warmup-steps",
        type=int,
        default=gfp.DEFAULT_GROUNDING_WARMUP_STEPS,
        help="Zero-action env steps after the teleport before the rgb is read. "
        "Raise if the teleported robot has not settled before the frame is read.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # This script exists only to render the start frame, so the perception
    # camera MUST initialize — Isaac Lab refuses to spawn a camera without
    # --enable_cameras. Force it on (like num_envs=1 below) rather than depend
    # on the operator remembering the flag.
    args.enable_cameras = True

    if not args.ground_start_frame:
        raise SystemExit(
            "render_grounded_mission_corpus.py requires --ground-start-frame: it "
            "exists only to supply the live render the grounding pass needs. For "
            "an ungrounded corpus, use the headless build_mission_corpus.py."
        )
    scene = _resolve_single_scene(args)
    args.scenes = [scene]

    # Resolve the scene USD before the multi-minute Kit boot so a typo fails in
    # milliseconds (scene_paths imports without Kit).
    from strafer_lab.tools.scene_paths import resolve_scene_usd_path

    try:
        scene_usd = resolve_scene_usd_path(scene=scene, search_root=args.scenes_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    # --- boot Isaac Sim once (all construction below is post-boot) ----------
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  (registers the isaaclab task suite)
    import strafer_lab.tasks  # noqa: F401  (registers the strafer tasks)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.scene.scene_geometry.spawn.usd_path = str(scene_usd)
    # One render per env.step: one physics tick (decimation=1) and a render every
    # tick (render_interval=1) so the warm-up step count maps directly to renders.
    env_cfg.decimation = 1
    env_cfg.sim.render_interval = 1
    print(f"[render_grounded] scene={scene} usd={scene_usd}")

    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    # The provider's app_update is the sole Kit pump per step; disable the
    # in-step KitVisualizer pump so the two do not double-render the viewport.
    env.unwrapped.render_enabled = False

    provider = gfp.make_render_grounding_frame_provider(
        env=env,
        app_update=simulation_app.update,
        spawn_z=args.spawn_z,
        warmup_steps=args.grounding_warmup_steps,
    )
    print(
        f"[render_grounded] live geometric grounding frame provider up "
        f"(spawn_z={args.spawn_z}, warmup_steps={args.grounding_warmup_steps})"
    )

    try:
        rc = build_mission_corpus.run(args, grounding_frame_provider=provider)
    finally:
        env.close()
        simulation_app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
