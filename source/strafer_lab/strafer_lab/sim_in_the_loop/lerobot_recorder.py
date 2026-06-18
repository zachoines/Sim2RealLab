"""LeRobot-backed :class:`EpisodeRecorder` for the bridge capture driver.

Adapts the harness's per-mission lifecycle onto the
:class:`strafer_lab.tools.lerobot_writer.StraferLeRobotWriter` episode
API (begin_episode → add_frame → end_episode), mapping executor terminal
states and hard-negative injection metadata onto the per-episode
extension columns. One mission = one episode.

Outcome mapping at ``end_episode``:

==================  =========================  ==========================
executor state      not injected               injected (actual mode set)
==================  =========================  ==========================
succeeded           outcome=succeeded,         outcome=<actual mode>,
                    category=on_course         category=<actual mode>,
                                               hard_negative=<actual mode>
failed / timeout    outcome=failed, category=on_course, no hard negative
cancelled/aborted   episode discarded — externally killed runs are
                    operational noise, not training signal
==================  =========================  ==========================

Injection columns (``injection_mode`` / ``injection_mode_actual`` /
``original_target_position_3d``) are recorded regardless of outcome —
they describe what was dispatched, not how it ended.

The module is pure Python: the writer is duck-typed so tests drive the
recorder with a fake. :class:`CmdVelGraceWatch` (the /cmd_vel-silence
abort signal for the harness) lives here too — it only needs two clock
callables.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from strafer_lab.sim_in_the_loop.harness import (
    EpisodeMeta,
    FrameBundle,
    MissionStatus,
)
from strafer_lab.sim_in_the_loop.mission import MissionSpec


# Externally-killed terminal states; episodes ending in one of these are
# discarded rather than labeled.
_DISCARD_STATES = frozenset({"cancelled", "canceled", "aborted", "discarded"})


def yaw_from_quat_xyzw(quat: tuple[float, float, float, float]) -> float:
    """Yaw (rad) from an ``(qx, qy, qz, qw)`` quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BridgeLeRobotRecorder:
    """Records harness episodes into a strafer LeRobot v3 dataset.

    Parameters
    ----------
    writer:
        A :class:`StraferLeRobotWriter` (or fake with the same episode
        API). The recorder reads ``writer.detections_max`` to decide
        whether to forward per-frame detections.
    scene_id:
        Recorded on every episode; resolves to the dataset's
        ``meta/scenes/<scene_id>/scene_metadata.json``.
    source_mission_source:
        Fallback when an episode's meta doesn't carry one.
    """

    def __init__(
        self,
        *,
        writer: Any,
        scene_id: str,
        source_driver: str = "bridge",
    ) -> None:
        self._writer = writer
        self._scene_id = scene_id
        self._source_driver = source_driver
        self._open_meta: EpisodeMeta | None = None
        self.episodes_kept = 0
        self.episodes_discarded = 0

    # ------------------------------------------------------------------
    # EpisodeRecorder protocol
    # ------------------------------------------------------------------

    def begin_episode(
        self, *, spec: MissionSpec, meta: EpisodeMeta, start_bundle: FrameBundle,
    ) -> None:
        start_xy_yaw = (
            float(start_bundle.robot_pos[0]),
            float(start_bundle.robot_pos[1]),
            yaw_from_quat_xyzw(start_bundle.robot_quat),
        )
        leg_initial_distance_m = None
        if meta.target_position_3d is not None:
            leg_initial_distance_m = math.hypot(
                meta.target_position_3d[0] - start_xy_yaw[0],
                meta.target_position_3d[1] - start_xy_yaw[1],
            )
        self._open_meta = meta
        self._writer.begin_episode(
            mission_text=meta.mission_text,
            scene_id=self._scene_id,
            target_label=meta.target_label,
            target_object_id=meta.target_object_id,
            target_position_3d=(
                list(meta.target_position_3d)
                if meta.target_position_3d is not None else None
            ),
            start_pose=list(start_xy_yaw),
            source_driver=self._source_driver,
            source_mission_source=meta.source_mission_source,
            paraphrases=list(meta.paraphrases),
            leg_initial_distance_m=leg_initial_distance_m,
            injection_mode=meta.injection_mode,
            generator_metadata=dict(meta.generator_metadata) or None,
        )

    def add_frame(self, bundle: FrameBundle) -> None:
        kwargs: dict[str, Any] = {
            "sim_time": float(bundle.sim_time_s),
            "pose": list(bundle.robot_pos) + list(bundle.robot_quat),
            "achieved_vel": list(bundle.achieved_vel),
            "action": list(bundle.action),
            "rgb_perception": bundle.rgb,
            "rgb_policy": bundle.rgb_policy,
            "depth_m": bundle.depth,
            "depth_policy_m": bundle.depth_policy,
        }
        if getattr(self._writer, "detections_max", None) is not None:
            kwargs["detections"] = list(bundle.detections or ())
        self._writer.add_frame(**kwargs)

    def end_episode(
        self, *, status: MissionStatus | None, discard: bool = False,
    ) -> None:
        meta = self._open_meta
        self._open_meta = None
        state = (status.state.lower() if status is not None else "")
        if discard or status is None or state in _DISCARD_STATES:
            self.episodes_discarded += 1
            self._writer.end_episode(discard=True)
            return

        injected = meta is not None and meta.injection_mode_actual is not None
        if status.succeeded and injected:
            outcome = outcome_category = hard_negative = meta.injection_mode_actual
        elif status.succeeded:
            outcome, outcome_category, hard_negative = "succeeded", "on_course", None
        else:
            # An honest mission that failed is still "on_course": the
            # outcome_category axis collapses honest success + honest failure
            # into on_course and reserves wrong_instance / wrong_room /
            # trajectory_violation for DELIBERATE misgrounding. The honest
            # failure is fully described by outcome="failed" alongside
            # outcome_category="on_course" — the pair, not either column alone.
            outcome, outcome_category, hard_negative = "failed", "on_course", None

        self.episodes_kept += 1
        self._writer.end_episode(
            outcome=outcome,
            outcome_category=outcome_category,
            hard_negative_category=hard_negative,
            injection_mode_actual=(meta.injection_mode_actual if meta else None),
            original_target_position_3d=(
                list(meta.original_target_position_3d)
                if meta is not None and meta.original_target_position_3d is not None
                else None
            ),
        )


class CmdVelGraceWatch:
    """Abort signal: /cmd_vel went silent mid-drive for longer than the grace.

    Armed at each episode start. The watch only fires after the episode
    has seen at least one /cmd_vel newer than its arm time — silence
    *before* the executor starts driving (planning, VLM grounding) is
    governed by the mission timeout, not by this watchdog.

    ``last_cmd_time`` and ``now`` must share a clock (monotonic seconds);
    the bridge wires them to
    :meth:`StraferAsyncPublisher.last_cmd_monotonic` and
    :func:`time.monotonic`.
    """

    REASON = "cmd_vel_silence"

    def __init__(
        self,
        *,
        last_cmd_time: Callable[[], float],
        grace_s: float,
        now: Callable[[], float],
    ) -> None:
        self._last_cmd_time = last_cmd_time
        self._grace_s = float(grace_s)
        self._now = now
        self._armed_at: float | None = None

    def arm(self) -> None:
        """Start a new episode window; prior /cmd_vel traffic is ignored."""
        self._armed_at = self._now()

    def check(self) -> str | None:
        """Return the abort reason when the grace is exceeded, else None."""
        if self._armed_at is None or self._grace_s <= 0:
            return None
        last_cmd = self._last_cmd_time()
        if last_cmd <= self._armed_at:
            return None  # executor has not started driving yet
        if self._now() - last_cmd > self._grace_s:
            return self.REASON
        return None
