"""Composable navigation environment configuration.

Expresses a Strafer navigation environment as a composition over the
orthogonal axes it actually varies on — **sensor stack**, **scene source**,
**realism level**, and **objective** — rather than one hand-written class per
populated cell of that matrix. A consumer asks for the combination it needs
(e.g. "RGB only, Infinigen, realistic") and the composition root materializes
the matching observation / action / event / scene managers from the shared
building blocks defined in :mod:`strafer_env_cfg`.

Four axis configurations drive the composition:

- :class:`SensorStackCfg` — ``cameras_required`` over the tokens
  ``rgb_full`` / ``depth_full`` (the 640x360 perception camera channels) and
  ``rgb_policy`` / ``depth_policy`` (the 80x60 policy camera channels). The
  selected tokens decide which camera prims the scene renders and which image
  observation terms the policy receives.
- :class:`SceneSourceCfg` — where the world geometry comes from (plane,
  Infinigen room USD, procedural primitive rooms, or none). This is the seam
  that lets a foreign USD become a parameter instead of a new env subclass.
- :class:`RealismCfg` — Ideal / Real / Robust domain-randomization and
  observation-noise tier.
- :class:`ObjectiveCfg` — what the command term asks the policy to do:
  converge on a fixed ``goal`` pose, or track a rolling ``subgoal`` along a
  sim-planned path. The objective selects the command / reward / termination
  blocks; observations are objective-agnostic because the goal-shaped terms
  read whatever the active command term emits.

The composition root selects among the *existing* MDP building blocks for the
requested axes, so an RL variant composed here is structurally identical to
the hand-written class it replaces — the observation tensor, action layout,
DR sequence, and scene a trained policy depends on are preserved exactly. The
selector keeps the capture-facing variants (teleop, bridge, scripted) free to
render a narrower or wider camera set than any RL variant, since those
consumers read camera data through scene handles rather than the policy
observation tensor.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from . import strafer_env_cfg as base

# Re-used building blocks (observation / action / event / command / reward /
# termination / curriculum / scene managers + runtime helpers). Selecting
# among these — rather than rebuilding them — is what guarantees a composed RL
# variant matches the class it replaces byte-for-byte.
from .strafer_env_cfg import (
    ActionsCfg_Ideal,
    ActionsCfg_Realistic,
    ActionsCfg_Robust,
    CommandsCfg,
    CommandsCfg_Infinigen,
    CommandsCfg_ProcRoom,
    CommandsCfg_ProcRoom_Subgoal,
    CurriculumCfg,
    CurriculumCfg_Infinigen,
    CurriculumCfg_ProcRoom,
    EventsCfg_Ideal,
    EventsCfg_Infinigen_Realistic,
    EventsCfg_Infinigen_Robust,
    EventsCfg_ProcRoom_Realistic,
    EventsCfg_ProcRoom_Robust,
    EventsCfg_Realistic,
    EventsCfg_Robust,
    ObsCfg_Depth_Ideal,
    ObsCfg_Depth_Realistic,
    ObsCfg_Depth_Robust,
    ObsCfg_Full_Ideal,
    ObsCfg_Full_Realistic,
    ObsCfg_Full_Robust,
    ObsCfg_NoCam_Ideal,
    ObsCfg_NoCam_Realistic,
    ObsCfg_NoCam_Robust,
    RewardsCfg,
    RewardsCfg_ProcRoom,
    RewardsCfg_ProcRoom_Subgoal,
    StraferSceneCfg,
    StraferSceneCfg_Infinigen,
    StraferSceneCfg_InfinigenPerception,
    StraferSceneCfg_NoCam,
    StraferSceneCfg_ProcRoom,
    StraferSceneCfg_ProcRoom_NoCam,
    TerminationsCfg,
    TerminationsCfg_ProcRoom,
    TerminationsCfg_ProcRoom_Subgoal,
    _apply_default_nav_runtime,
    _apply_infinigen_scene_setup,
    _apply_play_num_envs,
    _apply_procroom_physx_buffers,
    make_d555_camera_cfg,
    make_d555_perception_camera_cfg,
)

# ---------------------------------------------------------------------------
# Axis vocabularies
# ---------------------------------------------------------------------------

# Sensor-stack tokens. ``*_full`` ride the 640x360 perception camera
# (``d555_camera_perception``); ``*_policy`` ride the 80x60 policy camera
# (``d555_camera``). RGB tokens request the ``rgb`` channel, depth tokens the
# ``distance_to_image_plane`` channel.
SENSOR_TOKENS: tuple[str, ...] = ("rgb_full", "depth_full", "rgb_policy", "depth_policy")

_POLICY_TOKENS = ("rgb_policy", "depth_policy")
_PERCEPTION_TOKENS = ("rgb_full", "depth_full")

# Scene sources.
SCENE_SOURCES: tuple[str, ...] = ("plane", "infinigen", "procroom", "none")

# Realism tiers.
REALISM_LEVELS: tuple[str, ...] = ("ideal", "real", "robust")

# Objectives.
OBJECTIVE_KINDS: tuple[str, ...] = ("goal", "subgoal")


# ---------------------------------------------------------------------------
# Axis configurations
# ---------------------------------------------------------------------------


@configclass
class SensorStackCfg:
    """Which camera data products the environment exposes.

    ``cameras_required`` is the single parameter the env render set and the
    capture writer schema are both driven from, so the rendered cameras and
    the recorded columns cannot drift. An empty tuple is a camera-free
    (proprioceptive) environment.
    """

    cameras_required: tuple[str, ...] = ("depth_policy",)

    def normalized(self) -> tuple[str, ...]:
        """Return the requested tokens in canonical order, de-duplicated."""
        seen = set(self.cameras_required)
        unknown = seen - set(SENSOR_TOKENS)
        if unknown:
            raise ValueError(
                f"Unknown sensor token(s) {sorted(unknown)}; "
                f"valid tokens are {SENSOR_TOKENS}",
            )
        return tuple(t for t in SENSOR_TOKENS if t in seen)

    # -- policy camera (80x60) ------------------------------------------------

    def has_policy_camera(self) -> bool:
        return any(t in self.cameras_required for t in _POLICY_TOKENS)

    def policy_data_types(self) -> tuple[str, ...]:
        types: list[str] = []
        if "rgb_policy" in self.cameras_required:
            types.append("rgb")
        if "depth_policy" in self.cameras_required:
            types.append("distance_to_image_plane")
        return tuple(types)

    # -- perception camera (640x360) -----------------------------------------

    def has_perception_camera(self) -> bool:
        return any(t in self.cameras_required for t in _PERCEPTION_TOKENS)

    def perception_data_types(self) -> tuple[str, ...]:
        types: list[str] = []
        if "rgb_full" in self.cameras_required:
            types.append("rgb")
        if "depth_full" in self.cameras_required:
            types.append("distance_to_image_plane")
        return tuple(types)

    # -- observation profile --------------------------------------------------

    def obs_profile(self) -> str:
        """Map the policy-camera tokens to an observation profile.

        ``full`` (rgb+depth policy image obs), ``depth`` (depth policy image
        obs only), or ``nocam`` (proprioceptive only). RGB-only-policy stacks
        fall back to ``nocam`` for the observation tensor — the rgb_policy
        camera still renders for capture, there is just no rgb-only image
        observation term.
        """
        has_depth = "depth_policy" in self.cameras_required
        has_rgb = "rgb_policy" in self.cameras_required
        if has_depth and has_rgb:
            return "full"
        if has_depth:
            return "depth"
        return "nocam"


@configclass
class SceneSourceCfg:
    """Where the world geometry and metadata come from.

    ``kind`` selects the scene class + spawn/metadata loading + per-source
    runtime knobs. ``num_envs`` / ``env_spacing`` default to ``None`` so the
    per-source defaults apply; set them to override (e.g. a play preset).
    """

    kind: str = "procroom"
    num_envs: int | None = None
    env_spacing: float | None = None


@configclass
class RealismCfg:
    """Domain-randomization + observation-noise tier."""

    level: str = "real"


@configclass
class ObjectiveCfg:
    """What the command term asks the policy to do.

    ``goal`` keeps the established setup: a fixed goal pose per episode, the
    policy converges on it. ``subgoal`` swaps in the path-following task: a
    sim-planned path per episode, a rolling subgoal command, path-tracking
    rewards, and path-complete / off-path terminations. Observations are
    unchanged across objectives — the goal-shaped terms read whatever the
    active command term emits — so the obs/action contract is decided by the
    other axes alone.
    """

    kind: str = "goal"


# ---------------------------------------------------------------------------
# Selector tables — (axis) -> existing building block
# ---------------------------------------------------------------------------

_OBS_BY_REALISM_PROFILE = {
    ("ideal", "full"): ObsCfg_Full_Ideal,
    ("ideal", "depth"): ObsCfg_Depth_Ideal,
    ("ideal", "nocam"): ObsCfg_NoCam_Ideal,
    ("real", "full"): ObsCfg_Full_Realistic,
    ("real", "depth"): ObsCfg_Depth_Realistic,
    ("real", "nocam"): ObsCfg_NoCam_Realistic,
    ("robust", "full"): ObsCfg_Full_Robust,
    ("robust", "depth"): ObsCfg_Depth_Robust,
    ("robust", "nocam"): ObsCfg_NoCam_Robust,
}

_ACTIONS_BY_REALISM = {
    "ideal": ActionsCfg_Ideal,
    "real": ActionsCfg_Realistic,
    "robust": ActionsCfg_Robust,
}

# Events depend on BOTH the scene source (which structural reset/lift/room
# events run) and the realism level (how aggressive the DR is).
_EVENTS_BY_SOURCE_REALISM = {
    ("plane", "ideal"): EventsCfg_Ideal,
    ("plane", "real"): EventsCfg_Realistic,
    ("plane", "robust"): EventsCfg_Robust,
    ("infinigen", "real"): EventsCfg_Infinigen_Realistic,
    ("infinigen", "robust"): EventsCfg_Infinigen_Robust,
    ("procroom", "real"): EventsCfg_ProcRoom_Realistic,
    ("procroom", "robust"): EventsCfg_ProcRoom_Robust,
}

_COMMANDS_BY_SOURCE = {
    "plane": CommandsCfg,
    "infinigen": CommandsCfg_Infinigen,
    "procroom": CommandsCfg_ProcRoom,
    "none": CommandsCfg,
}

_REWARDS_BY_SOURCE = {
    "plane": RewardsCfg,
    "infinigen": RewardsCfg,
    "procroom": RewardsCfg_ProcRoom,
    "none": RewardsCfg,
}

_TERMINATIONS_BY_SOURCE = {
    "plane": TerminationsCfg,
    "infinigen": TerminationsCfg,
    "procroom": TerminationsCfg_ProcRoom,
    "none": TerminationsCfg,
}

# The subgoal objective swaps the command / reward / termination blocks and
# is composed for the ProcRoom source only: its planner consumes the
# occupancy grids the procedural-room generator builds. Extending it to
# another source means giving that source a free-space grid first.
_COMMANDS_BY_SOURCE_SUBGOAL = {
    "procroom": CommandsCfg_ProcRoom_Subgoal,
}

_REWARDS_BY_SOURCE_SUBGOAL = {
    "procroom": RewardsCfg_ProcRoom_Subgoal,
}

_TERMINATIONS_BY_SOURCE_SUBGOAL = {
    "procroom": TerminationsCfg_ProcRoom_Subgoal,
}

_CURRICULUM_BY_SOURCE = {
    "plane": CurriculumCfg,
    "infinigen": CurriculumCfg_Infinigen,
    "procroom": CurriculumCfg_ProcRoom,
    "none": CurriculumCfg,
}


def _select_scene(scene_source: SceneSourceCfg, sensors: SensorStackCfg):
    """Pick + size the scene class for the requested source and sensor stack.

    For RL sources (plane / procroom / depth-only infinigen) this returns the
    same scene class the hand-written variant used, so the composed env is
    scene-identical. The Infinigen perception scene (both cameras) backs the
    capture variants and is pruned to the requested stack afterwards.
    """
    kind = scene_source.kind
    has_policy_cam = sensors.has_policy_camera()
    has_perception_cam = sensors.has_perception_camera()

    if kind == "plane":
        if has_policy_cam:
            scene = StraferSceneCfg(
                num_envs=base._STANDARD_TRAIN_NUM_ENVS,
                env_spacing=base._STANDARD_ENV_SPACING,
            )
        else:
            scene = StraferSceneCfg_NoCam(
                num_envs=base._STANDARD_TRAIN_NUM_ENVS,
                env_spacing=base._STANDARD_ENV_SPACING,
            )
    elif kind == "procroom":
        if has_policy_cam:
            scene = StraferSceneCfg_ProcRoom(
                num_envs=base._PROCROOM_DEPTH_TRAIN_NUM_ENVS,
                env_spacing=base._PROCROOM_ENV_SPACING,
            )
        else:
            scene = StraferSceneCfg_ProcRoom_NoCam(
                num_envs=base._PROCROOM_NOCAM_TRAIN_NUM_ENVS,
                env_spacing=base._PROCROOM_ENV_SPACING,
            )
    elif kind == "infinigen":
        if has_perception_cam:
            scene = StraferSceneCfg_InfinigenPerception(
                num_envs=base._INFINIGEN_PERCEPTION_TRAIN_NUM_ENVS,
                env_spacing=0.0,
            )
        else:
            scene = StraferSceneCfg_Infinigen(
                num_envs=base._INFINIGEN_TRAIN_NUM_ENVS,
                env_spacing=0.0,
            )
    elif kind == "none":
        scene = StraferSceneCfg_NoCam(
            num_envs=base._STANDARD_TRAIN_NUM_ENVS,
            env_spacing=base._STANDARD_ENV_SPACING,
        )
    else:
        raise ValueError(f"Unknown scene source {kind!r}; valid: {SCENE_SOURCES}")

    if scene_source.num_envs is not None:
        scene.num_envs = scene_source.num_envs
    if scene_source.env_spacing is not None:
        scene.env_spacing = scene_source.env_spacing
    return scene


def _prune_scene_cameras(scene, sensors: SensorStackCfg) -> None:
    """Trim the scene's cameras to the requested sensor stack.

    The policy camera always keeps an ``rgb`` channel, even for a depth-only
    policy: the RTX viewport / ``--video`` colour pipeline needs an rgb render
    product to come up, independent of what the observation reads.
    """
    # Policy camera (80x60): rgb (for the viewport) unioned with observed channels.
    if sensors.has_policy_camera():
        if hasattr(scene, "d555_camera"):
            data_types = sensors.policy_data_types()
            if "rgb" not in data_types:
                data_types = ("rgb",) + data_types
            scene.d555_camera = make_d555_camera_cfg(data_types=data_types)
    elif hasattr(scene, "d555_camera"):
        scene.d555_camera = None

    # Perception camera (640x360).
    if sensors.has_perception_camera():
        if hasattr(scene, "d555_camera_perception"):
            cam = make_d555_perception_camera_cfg()
            cam.data_types = list(sensors.perception_data_types())
            scene.d555_camera_perception = cam
    elif hasattr(scene, "d555_camera_perception"):
        scene.d555_camera_perception = None


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------


@configclass
class _ComposedStraferNavEnvCfg(base._BaseStraferNavEnvCfg):
    """Navigation env composed from sensor / scene-source / realism axes.

    Subclasses set the three axis attributes (and optionally
    ``play_num_envs``); ``__post_init__`` materializes the standard manager
    cfgs from them by selecting the matching shared building blocks.
    """

    sensors: SensorStackCfg = SensorStackCfg()
    scene_source: SceneSourceCfg = SceneSourceCfg()
    realism: RealismCfg = RealismCfg()
    objective: ObjectiveCfg = ObjectiveCfg()

    # When set, the env shrinks its scene to this many envs (play/eval preset).
    play_num_envs: int | None = None

    def __post_init__(self):
        sensors = self.sensors
        source = self.scene_source
        level = self.realism.level
        if level not in REALISM_LEVELS:
            raise ValueError(f"Unknown realism {level!r}; valid: {REALISM_LEVELS}")
        objective = self.objective.kind
        if objective not in OBJECTIVE_KINDS:
            raise ValueError(
                f"Unknown objective {objective!r}; valid: {OBJECTIVE_KINDS}"
            )
        profile = sensors.obs_profile()
        kind = source.kind

        # --- Scene ---
        scene = _select_scene(source, sensors)
        _prune_scene_cameras(scene, sensors)
        self.scene = scene

        # --- Managers (select shared building blocks) ---
        self.observations = _OBS_BY_REALISM_PROFILE[(level, profile)]()
        self.actions = _ACTIONS_BY_REALISM[level]()
        try:
            self.events = _EVENTS_BY_SOURCE_REALISM[(kind, level)]()
        except KeyError as exc:
            raise ValueError(
                f"No event tier for scene_source={kind!r}, realism={level!r}",
            ) from exc
        if objective == "subgoal":
            try:
                self.commands = _COMMANDS_BY_SOURCE_SUBGOAL[kind]()
                self.rewards = _REWARDS_BY_SOURCE_SUBGOAL[kind]()
                self.terminations = _TERMINATIONS_BY_SOURCE_SUBGOAL[kind]()
            except KeyError as exc:
                raise ValueError(
                    f"The subgoal objective is not composed for "
                    f"scene_source={kind!r}; valid: "
                    f"{sorted(_COMMANDS_BY_SOURCE_SUBGOAL)}",
                ) from exc
            # Goal-pose reset noise targets the fixed-goal command's state;
            # the subgoal task's deployment-noise analog is the planner's
            # per-waypoint perturbation, configured on the command term.
            if hasattr(self.events, "randomize_goal_noise"):
                self.events.randomize_goal_noise = None
            # Robust tier additionally randomizes the lookahead distance so the
            # policy is not brittle to the deployed selector's exact lookahead;
            # the realistic baseline trains at the fixed deployment distance.
            if level == "robust":
                self.commands.goal_command.lookahead_randomization_m = (
                    base._SUBGOAL_ROBUST_LOOKAHEAD_BAND
                )
        else:
            self.commands = _COMMANDS_BY_SOURCE[kind]()
            self.rewards = _REWARDS_BY_SOURCE[kind]()
            self.terminations = _TERMINATIONS_BY_SOURCE[kind]()
        self.curriculum = _CURRICULUM_BY_SOURCE[kind]()

        # --- Shared runtime defaults (dt / decimation / render / episode) ---
        _apply_default_nav_runtime(self)

        # --- Per-source structural setup ---
        if kind == "infinigen":
            _apply_infinigen_scene_setup(self)
        elif kind == "procroom":
            _apply_procroom_physx_buffers(self)

        # --- Play/eval shrink ---
        if self.play_num_envs is not None:
            _apply_play_num_envs(self, num_envs=self.play_num_envs)


# ---------------------------------------------------------------------------
# Named composed variants
#
# RL variants (fixed sensor stack — the obs contract a trained policy was
# fitted against). Each is structurally identical to the hand-written class
# it replaces; the snapshot contract test asserts the byte-level identity.
# ---------------------------------------------------------------------------


@configclass
class StraferNavCfg_RLDepth_Real(_ComposedStraferNavEnvCfg):
    """RL depth training (ProcRoom, realistic). Replaces the realistic
    ProcRoom depth variant; obs/action/DR preserved byte-identical."""

    sensors = SensorStackCfg(cameras_required=("depth_policy",))
    scene_source = SceneSourceCfg(kind="procroom")
    realism = RealismCfg(level="real")


@configclass
class StraferNavCfg_RLDepth_Real_PLAY(StraferNavCfg_RLDepth_Real):
    """Play/eval preset for RL depth (realistic)."""

    play_num_envs = base._PROCROOM_DEPTH_PLAY_NUM_ENVS


@configclass
class StraferNavCfg_RLDepth_Robust(_ComposedStraferNavEnvCfg):
    """RL depth training (ProcRoom, robust DR). Replaces the robust ProcRoom
    depth variant; obs/action/DR preserved byte-identical."""

    sensors = SensorStackCfg(cameras_required=("depth_policy",))
    scene_source = SceneSourceCfg(kind="procroom")
    realism = RealismCfg(level="robust")


@configclass
class StraferNavCfg_RLDepth_Robust_PLAY(StraferNavCfg_RLDepth_Robust):
    """Play/eval preset for RL depth (robust)."""

    play_num_envs = base._PROCROOM_DEPTH_PLAY_NUM_ENVS


@configclass
class StraferNavCfg_RLNoCam(_ComposedStraferNavEnvCfg):
    """RL proprioceptive baseline (ProcRoom, realistic). Replaces the
    realistic ProcRoom NoCam variant; obs/action/DR preserved byte-identical."""

    sensors = SensorStackCfg(cameras_required=())
    scene_source = SceneSourceCfg(kind="procroom")
    realism = RealismCfg(level="real")


@configclass
class StraferNavCfg_RLNoCam_PLAY(StraferNavCfg_RLNoCam):
    """Play/eval preset for RL no-cam baseline."""

    play_num_envs = base._PROCROOM_NOCAM_PLAY_NUM_ENVS


@configclass
class StraferNavCfg_RLNoCamSubgoal_Real(_ComposedStraferNavEnvCfg):
    """RL proprioceptive subgoal tracking (ProcRoom, realistic). Same obs/
    action contract as the no-cam baseline; the goal-shaped observation
    fields refer to a rolling subgoal on a sim-planned path."""

    sensors = SensorStackCfg(cameras_required=())
    scene_source = SceneSourceCfg(kind="procroom")
    realism = RealismCfg(level="real")
    objective = ObjectiveCfg(kind="subgoal")


@configclass
class StraferNavCfg_RLNoCamSubgoal_Real_PLAY(StraferNavCfg_RLNoCamSubgoal_Real):
    """Play/eval preset for RL no-cam subgoal tracking (realistic)."""

    play_num_envs = base._PROCROOM_NOCAM_PLAY_NUM_ENVS


@configclass
class StraferNavCfg_RLNoCamSubgoal_Robust(_ComposedStraferNavEnvCfg):
    """RL proprioceptive subgoal tracking (ProcRoom, robust DR)."""

    sensors = SensorStackCfg(cameras_required=())
    scene_source = SceneSourceCfg(kind="procroom")
    realism = RealismCfg(level="robust")
    objective = ObjectiveCfg(kind="subgoal")


@configclass
class StraferNavCfg_RLNoCamSubgoal_Robust_PLAY(StraferNavCfg_RLNoCamSubgoal_Robust):
    """Play/eval preset for RL no-cam subgoal tracking (robust)."""

    play_num_envs = base._PROCROOM_NOCAM_PLAY_NUM_ENVS


# ---------------------------------------------------------------------------
# Capture variants (operator-selectable stack — NOT snapshot-gated). These
# read camera data through scene handles, not the policy obs tensor, so their
# render set is free to differ from any RL variant. The ``cameras_required``
# defaults below are the per-consumer presets; capture.py --sensors overrides.
# ---------------------------------------------------------------------------


@configclass
class StraferNavCfg_TeleopCapture(_ComposedStraferNavEnvCfg):
    """Teleop capture (Infinigen, realistic). Default reduced stack = full-res
    RGB only — all CLIP / grounding-VLM finetune needs. ``--sensors``
    expands it (e.g. +depth_full) per session for VLA capture."""

    sensors = SensorStackCfg(cameras_required=("rgb_full",))
    scene_source = SceneSourceCfg(kind="infinigen")
    realism = RealismCfg(level="real")


@configclass
class StraferNavCfg_BridgeAutonomy(_ComposedStraferNavEnvCfg):
    """Bridge sim-in-the-loop (Infinigen, realistic). Default expanded stack —
    full RGB + full depth + policy depth — so the harness captures the full
    data product the autonomy stack and downstream training consume."""

    sensors = SensorStackCfg(
        cameras_required=("rgb_full", "depth_full", "depth_policy"),
    )
    scene_source = SceneSourceCfg(kind="infinigen")
    realism = RealismCfg(level="real")


@configclass
class StraferNavCfg_Coverage(_ComposedStraferNavEnvCfg):
    """Scripted coverage capture (Infinigen, realistic). Default stack =
    full RGB + full depth; ``--sensors`` overrides."""

    sensors = SensorStackCfg(cameras_required=("rgb_full", "depth_full"))
    scene_source = SceneSourceCfg(kind="infinigen")
    realism = RealismCfg(level="real")


# ---------------------------------------------------------------------------
# Minimal plane bases (not registered). These compose the simplest scaffold —
# plane scene, no domain randomization — for the MDP-component unit tests,
# which build one of these and then override the manager under test.
# ---------------------------------------------------------------------------


@configclass
class StraferNavCfg_NoCam_Ideal(_ComposedStraferNavEnvCfg):
    """Proprioceptive plane env, no DR. Unit-test scaffold."""

    sensors = SensorStackCfg(cameras_required=())
    scene_source = SceneSourceCfg(kind="plane")
    realism = RealismCfg(level="ideal")


@configclass
class StraferNavCfg_Depth_Ideal(_ComposedStraferNavEnvCfg):
    """Depth-policy plane env, no DR. Unit-test scaffold."""

    sensors = SensorStackCfg(cameras_required=("depth_policy",))
    scene_source = SceneSourceCfg(kind="plane")
    realism = RealismCfg(level="ideal")
