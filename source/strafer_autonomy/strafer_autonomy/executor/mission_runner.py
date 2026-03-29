"""Concrete Jetson-side mission runner for bounded autonomy skill execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from threading import Event, Lock, Thread
import time
from typing import Any

from strafer_autonomy.clients import GroundingClient, PlannerClient, RosClient
from strafer_autonomy.executor.command_server import (
    MissionCommandHandler,
    MissionStatusSnapshot,
    MissionSubmission,
)
from strafer_autonomy.schemas import (
    GoalPoseCandidate,
    GroundingRequest,
    GroundingResult,
    MissionPlan,
    PlannerRequest,
    Pose3D,
    SceneObservation,
    SkillCall,
    SkillResult,
)


DEFAULT_AVAILABLE_SKILLS = (
    "capture_scene_observation",
    "locate_semantic_target",
    "scan_for_target",
    "describe_scene",
    "project_detection_to_goal_pose",
    "navigate_to_pose",
    # "orient_relative_to_target" — deferred from MVP; handler kept for post-MVP.
    "wait",
    "cancel_mission",
    "report_status",
)


@dataclass(frozen=True)
class MissionRunnerConfig:
    """Executor runtime settings for mission planning and step dispatch."""

    available_skills: tuple[str, ...] = DEFAULT_AVAILABLE_SKILLS
    replace_join_timeout_s: float = 5.0
    wait_poll_period_s: float = 0.1
    default_grounding_max_image_side: int = 1024
    default_standoff_m: float = 0.7
    default_navigation_timeout_s: float = 90.0
    default_navigation_backend: str = "nav2"


@dataclass
class _MissionRuntime:
    """Mutable state for one active or recently completed mission."""

    mission_id: str
    request_id: str
    raw_command: str
    source: str
    started_at: float
    active: bool = True
    state: str = "planning"
    current_step_id: str = ""
    current_skill: str = ""
    message: str = "Mission accepted."
    error_code: str = ""
    cancel_event: Event = field(default_factory=Event)
    resume_event: Event = field(default_factory=Event)
    thread: Thread | None = None
    plan: MissionPlan | None = None
    latest_observation: SceneObservation | None = None
    latest_grounding: GroundingResult | None = None
    latest_goal_pose: GoalPoseCandidate | None = None
    step_results: list[SkillResult] = field(default_factory=list)


class MissionRunner(MissionCommandHandler):
    """Concrete mission handler that sequences planner, VLM, and ROS skills."""

    def __init__(
        self,
        *,
        planner_client: PlannerClient,
        grounding_client: GroundingClient,
        ros_client: RosClient,
        config: MissionRunnerConfig | None = None,
    ) -> None:
        self._planner_client = planner_client
        self._grounding_client = grounding_client
        self._ros_client = ros_client
        self._config = config or MissionRunnerConfig()
        self._lock = Lock()
        self._active_runtime: _MissionRuntime | None = None
        self._last_runtime: _MissionRuntime | None = None

    @property
    def config(self) -> MissionRunnerConfig:
        """Return immutable mission-runner configuration."""

        return self._config

    def start_mission(
        self,
        *,
        request_id: str,
        raw_command: str,
        source: str,
        replace_active_mission: bool,
    ) -> MissionSubmission:
        """Accept a new mission and start executing it in the background."""

        thread_to_join: Thread | None = None
        should_cancel_robot_actions = False
        with self._lock:
            active_runtime = self._active_runtime
            if active_runtime is not None and active_runtime.active:
                if not replace_active_mission:
                    return MissionSubmission(
                        accepted=False,
                        mission_id=active_runtime.mission_id,
                        final_state=active_runtime.state,
                        error_code="mission_active",
                    message="A mission is already running. Re-submit with replace_active_mission=true.",
                    )
                self._request_cancel_locked(
                    active_runtime,
                    state="canceling",
                    message="Mission replacement requested.",
                    error_code="mission_replaced",
                )
                thread_to_join = active_runtime.thread
                should_cancel_robot_actions = True

        if should_cancel_robot_actions:
            self._cancel_robot_actions()

        if thread_to_join is not None:
            thread_to_join.join(timeout=self._config.replace_join_timeout_s)
            if thread_to_join.is_alive():
                snapshot = self.get_status()
                return MissionSubmission(
                    accepted=False,
                    mission_id=snapshot.mission_id,
                    final_state=snapshot.state,
                    error_code="mission_replace_timeout",
                    message="Active mission did not stop within the replacement timeout.",
                )

        runtime = _MissionRuntime(
            mission_id=request_id,
            request_id=request_id,
            raw_command=raw_command,
            source=source,
            started_at=time.time(),
        )
        worker = Thread(
            target=self._run_mission,
            args=(runtime,),
            name=f"mission-{runtime.mission_id}",
            daemon=True,
        )
        runtime.thread = worker
        with self._lock:
            self._active_runtime = runtime
            self._last_runtime = runtime
        worker.start()
        return MissionSubmission(
            accepted=True,
            mission_id=runtime.mission_id,
            message="Mission accepted.",
        )

    def get_status(self) -> MissionStatusSnapshot:
        """Return the current active mission snapshot or the most recent final state."""

        with self._lock:
            runtime = self._active_runtime or self._last_runtime
            if runtime is None:
                return MissionStatusSnapshot(active=False)
            return self._snapshot_from_runtime_locked(runtime)

    def cancel_active_mission(self) -> MissionStatusSnapshot:
        """Request cancellation of the current mission and return the latest snapshot."""

        should_cancel_robot_actions = False
        with self._lock:
            runtime = self._active_runtime
            if runtime is None or not runtime.active:
                if self._last_runtime is None:
                    return MissionStatusSnapshot(
                        active=False,
                        state="idle",
                        message="No mission has been started yet.",
                    )
                return self._snapshot_from_runtime_locked(self._last_runtime)
            self._request_cancel_locked(
                runtime,
                state="canceling",
                message="Cancellation requested.",
                error_code="mission_canceled",
            )
            snapshot = self._snapshot_from_runtime_locked(runtime)
            should_cancel_robot_actions = True

        if should_cancel_robot_actions:
            self._cancel_robot_actions()
        return snapshot

    def _run_mission(self, runtime: _MissionRuntime) -> None:
        try:
            self._set_runtime_state(runtime, state="planning", message="Requesting plan from planner service.")
            robot_state = self._safe_get_robot_state()
            plan = self._planner_client.plan_mission(
                PlannerRequest(
                    request_id=runtime.request_id,
                    raw_command=runtime.raw_command,
                    robot_state=robot_state,
                    active_mission_summary=self._active_mission_summary(runtime),
                    available_skills=self._config.available_skills,
                )
            )
            validation_errors = self._validate_plan(plan)
            if validation_errors:
                self._finish_runtime(
                    runtime,
                    state="failed",
                    message=f"Plan validation failed: {'; '.join(validation_errors)}",
                    error_code="plan_validation_failed",
                )
                return

            with self._lock:
                runtime.plan = plan
                runtime.mission_id = plan.mission_id
                runtime.message = f"Planner returned {len(plan.steps)} steps."
                runtime.error_code = ""

            for step in plan.steps:
                if runtime.cancel_event.is_set():
                    self._finish_runtime(
                        runtime,
                        state="canceled",
                        message="Mission canceled before step execution.",
                        error_code="mission_canceled",
                    )
                    return

                result = self._execute_step_with_retries(runtime, step)
                with self._lock:
                    runtime.step_results.append(result)

                if result.status == "succeeded":
                    continue
                if result.status == "canceled":
                    self._finish_runtime(
                        runtime,
                        state="canceled",
                        message=result.message or f"Skill '{step.skill}' canceled the mission.",
                        error_code=result.error_code or "mission_canceled",
                    )
                    return
                if result.status == "timeout":
                    self._finish_runtime(
                        runtime,
                        state="timeout",
                        message=result.message or f"Skill '{step.skill}' timed out.",
                        error_code=result.error_code or "skill_timeout",
                    )
                    return
                self._finish_runtime(
                    runtime,
                    state="failed",
                    message=result.message or f"Skill '{step.skill}' failed.",
                    error_code=result.error_code or "skill_failed",
                )
                return

            self._finish_runtime(
                runtime,
                state="succeeded",
                message="Mission completed successfully.",
                error_code="",
            )
        except Exception as exc:
            self._finish_runtime(
                runtime,
                state="failed",
                message=f"Mission execution raised an exception: {exc}",
                error_code="mission_exception",
            )

    # ------------------------------------------------------------------
    # Plan validation
    # ------------------------------------------------------------------

    _RECOGNISED_BACKENDS = frozenset({"nav2", "direct"})

    def _validate_plan(self, plan: MissionPlan) -> list[str]:
        """Validate a planner-returned mission plan before execution.

        Returns a list of human-readable error strings.  An empty list means
        the plan is valid and safe to execute.
        """
        errors: list[str] = []
        allowed = set(self._config.available_skills)
        seen_ids: set[str] = set()
        for step in plan.steps:
            if step.skill not in allowed:
                errors.append(f"Unknown skill '{step.skill}' (step {step.step_id})")
            if step.step_id in seen_ids:
                errors.append(f"Duplicate step_id '{step.step_id}'")
            seen_ids.add(step.step_id)
            backend = step.args.get("execution_backend")
            if backend is not None and backend not in self._RECOGNISED_BACKENDS:
                errors.append(
                    f"Unrecognised execution_backend '{backend}' (step {step.step_id})"
                )
        return errors

    def _execute_step_with_retries(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        """Execute a skill step, retrying up to ``step.retry_limit`` times on failure."""

        max_attempts = step.retry_limit + 1
        for attempt in range(1, max_attempts + 1):
            if runtime.cancel_event.is_set():
                return SkillResult(
                    step_id=step.step_id,
                    skill=step.skill,
                    status="canceled",
                    outputs={},
                    message="Mission canceled before retry.",
                    error_code="mission_canceled",
                    started_at=time.time(),
                    finished_at=time.time(),
                )

            self._set_runtime_state(
                runtime,
                state="executing",
                current_step_id=step.step_id,
                current_skill=step.skill,
                message=(
                    f"Executing skill '{step.skill}'."
                    if attempt == 1
                    else f"Retrying skill '{step.skill}' (attempt {attempt}/{max_attempts})."
                ),
                error_code="",
            )
            result = self._execute_step(runtime, step)

            if result.status == "succeeded" or result.status == "canceled":
                return result
            if attempt < max_attempts:
                continue
            return result

        # Unreachable, but keeps the type checker happy.
        return result  # type: ignore[possibly-undefined]

    def _execute_step(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        if step.skill == "capture_scene_observation":
            return self._capture_scene_observation(runtime, step)
        if step.skill == "locate_semantic_target":
            return self._locate_semantic_target(runtime, step)
        if step.skill == "scan_for_target":
            return self._scan_for_target(runtime, step)
        if step.skill == "describe_scene":
            return self._describe_scene(runtime, step)
        if step.skill == "project_detection_to_goal_pose":
            return self._project_detection_to_goal_pose(runtime, step)
        if step.skill == "navigate_to_pose":
            return self._navigate_to_pose(runtime, step)
        if step.skill == "orient_relative_to_target":
            return self._orient_relative_to_target(runtime, step)
        if step.skill == "wait":
            return self._wait(runtime, step)
        if step.skill == "report_status":
            return self._report_status(step)
        if step.skill == "cancel_mission":
            return self._cancel_step(runtime, step)
        return self._failed_result(
            step,
            message=f"Unsupported skill '{step.skill}'.",
            error_code="unsupported_skill",
        )

    def _capture_scene_observation(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        try:
            observation = self._ros_client.capture_scene_observation()
            with self._lock:
                runtime.latest_observation = observation
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs={
                    "observation_id": observation.observation_id,
                    "stamp_sec": observation.stamp_sec,
                    "camera_frame": observation.camera_frame,
                },
                message="Captured scene observation.",
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            return self._failed_result(step, f"Failed to capture scene observation: {exc}", "capture_failed", started_at)

    def _locate_semantic_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        label = str(step.args.get("label", "")).strip()
        if not label:
            return self._failed_result(step, "Grounding step is missing 'label'.", "invalid_args", started_at)

        observation = runtime.latest_observation
        if observation is None:
            try:
                observation = self._ros_client.capture_scene_observation()
                with self._lock:
                    runtime.latest_observation = observation
            except Exception as exc:
                return self._failed_result(
                    step,
                    f"Grounding could not capture a fresh observation: {exc}",
                    "capture_failed",
                    started_at,
                )
        assert observation is not None

        prompt = str(step.args.get("prompt") or f"Locate: {label}")
        try:
            grounding = self._grounding_client.locate_semantic_target(
                GroundingRequest(
                    request_id=f"{runtime.mission_id}:{step.step_id}",
                    prompt=prompt,
                    image_rgb_u8=self._bgr_to_rgb(observation.color_image_bgr),
                    image_stamp_sec=observation.stamp_sec,
                    max_image_side=int(
                        step.args.get("max_image_side", self._config.default_grounding_max_image_side)
                    ),
                    return_debug_overlay=bool(step.args.get("return_debug_overlay", False)),
                )
            )
            with self._lock:
                runtime.latest_grounding = grounding

            if not grounding.found or grounding.bbox_2d is None:
                return self._failed_result(
                    step,
                    message=f"Target '{label}' was not grounded in the current observation.",
                    error_code="target_not_found",
                    started_at=started_at,
                    outputs=self._grounding_outputs(grounding),
                )
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs=self._grounding_outputs(grounding),
                message=f"Grounded target '{label}'.",
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            return self._failed_result(step, f"Grounding request failed: {exc}", "grounding_failed", started_at)

    def _scan_for_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        """Rotate-and-ground loop: scan headings until the target is found.

        Args passed through ``step.args``:
            label (str): target to search for (required)
            max_scan_steps (int): number of rotation increments (default 6)
            scan_arc_deg (float): total arc to sweep in degrees (default 360)
            prompt (str | None): optional VLM prompt override
            max_image_side (int): image resize limit (default from config)
        """
        import math

        started_at = time.time()
        label = str(step.args.get("label", "")).strip()
        if not label:
            return self._failed_result(step, "scan_for_target is missing 'label'.", "invalid_args", started_at)

        max_scan_steps = int(step.args.get("max_scan_steps", 6))
        scan_arc_deg = float(step.args.get("scan_arc_deg", 360))
        step_angle_rad = scan_arc_deg * math.pi / 180.0 / max_scan_steps
        prompt = str(step.args.get("prompt") or f"Locate: {label}")
        max_image_side = int(step.args.get("max_image_side", self._config.default_grounding_max_image_side))

        for i in range(max_scan_steps):
            # a. capture scene observation
            try:
                observation = self._ros_client.capture_scene_observation()
                with self._lock:
                    runtime.latest_observation = observation
            except Exception as exc:
                return self._failed_result(
                    step, f"Scan capture failed at heading {i}: {exc}", "capture_failed", started_at,
                )

            # b. attempt grounding
            try:
                grounding = self._grounding_client.locate_semantic_target(
                    GroundingRequest(
                        request_id=f"{runtime.mission_id}:{step.step_id}:scan_{i}",
                        prompt=prompt,
                        image_rgb_u8=self._bgr_to_rgb(observation.color_image_bgr),
                        image_stamp_sec=observation.stamp_sec,
                        max_image_side=max_image_side,
                    )
                )
                with self._lock:
                    runtime.latest_grounding = grounding

                if grounding.found and grounding.bbox_2d is not None:
                    return SkillResult(
                        step_id=step.step_id,
                        skill=step.skill,
                        status="succeeded",
                        outputs={
                            "heading_index": i,
                            **self._grounding_outputs(grounding),
                        },
                        message=f"Target '{label}' found at heading {i}.",
                        started_at=started_at,
                        finished_at=time.time(),
                    )
            except Exception as exc:
                return self._failed_result(
                    step, f"Grounding failed during scan at heading {i}: {exc}", "grounding_failed", started_at,
                )

            # c. check cancel
            if runtime.cancel_event.is_set():
                return SkillResult(
                    step_id=step.step_id,
                    skill=step.skill,
                    status="canceled",
                    outputs={"heading_index": i},
                    error_code="mission_canceled",
                    message="Scan canceled.",
                    started_at=started_at,
                    finished_at=time.time(),
                )

            # d. rotate to next heading (skip after last heading)
            if i < max_scan_steps - 1:
                try:
                    rotate_result = self._ros_client.rotate_in_place(
                        step_id=f"{step.step_id}:rotate_{i}",
                        yaw_delta_rad=step_angle_rad,
                    )
                    if rotate_result.status != "succeeded":
                        return self._failed_result(
                            step,
                            f"Rotation failed at heading {i}: {rotate_result.message}",
                            rotate_result.error_code or "rotation_failed",
                            started_at,
                        )
                except Exception as exc:
                    return self._failed_result(
                        step, f"Rotation failed at heading {i}: {exc}", "rotation_failed", started_at,
                    )

        # Exhausted all headings without finding target
        return self._failed_result(
            step,
            message=f"Target '{label}' not found after full {scan_arc_deg:.0f}° scan.",
            error_code="target_not_found_after_scan",
            started_at=started_at,
            outputs={"headings_checked": max_scan_steps},
        )

    def _describe_scene(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        """Capture (or reuse) an observation and return a VLM scene description."""
        started_at = time.time()

        observation = runtime.latest_observation
        if observation is None:
            try:
                observation = self._ros_client.capture_scene_observation()
                with self._lock:
                    runtime.latest_observation = observation
            except Exception as exc:
                return self._failed_result(
                    step, f"Failed to capture observation for description: {exc}", "capture_failed", started_at,
                )

        prompt = step.args.get("prompt")
        max_image_side = int(step.args.get("max_image_side", self._config.default_grounding_max_image_side))

        try:
            desc = self._grounding_client.describe_scene(
                request_id=f"{runtime.mission_id}:{step.step_id}",
                image_rgb_u8=self._bgr_to_rgb(observation.color_image_bgr),
                prompt=prompt,
                max_image_side=max_image_side,
            )
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs={"description": desc.description, "latency_s": desc.latency_s},
                message="Scene described.",
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            return self._failed_result(step, f"Scene description failed: {exc}", "describe_failed", started_at)

    def _project_detection_to_goal_pose(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        observation = runtime.latest_observation
        grounding = runtime.latest_grounding
        if observation is None or grounding is None or grounding.bbox_2d is None:
            return self._failed_result(
                step,
                "Projection requires a prior observation and successful grounding result.",
                "projection_prereq_missing",
                started_at,
            )

        try:
            candidate = self._ros_client.project_detection_to_goal_pose(
                request_id=f"{runtime.mission_id}:{step.step_id}",
                image_stamp_sec=observation.stamp_sec,
                bbox_2d=grounding.bbox_2d,
                standoff_m=float(step.args.get("standoff_m", self._config.default_standoff_m)),
                target_label=grounding.label or str(step.args.get("target_label", "")) or None,
            )
            with self._lock:
                runtime.latest_goal_pose = candidate

            if not candidate.found or candidate.goal_pose is None:
                return self._failed_result(
                    step,
                    message=candidate.message or "Projection did not produce a reachable goal pose.",
                    error_code="goal_projection_failed",
                    started_at=started_at,
                    outputs=self._goal_pose_outputs(candidate),
                )
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs=self._goal_pose_outputs(candidate),
                message=candidate.message or "Projected goal pose successfully.",
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            return self._failed_result(step, f"Goal projection failed: {exc}", "goal_projection_failed", started_at)

    def _navigate_to_pose(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        goal_source = str(step.args.get("goal_source", "projected_target"))
        goal_pose: Pose3D | None
        if goal_source == "projected_target":
            candidate = runtime.latest_goal_pose
            goal_pose = candidate.goal_pose if candidate is not None else None
        else:
            goal_pose = self._parse_pose3d(step.args.get("goal_pose"))

        if goal_pose is None:
            return self._failed_result(
                step,
                "Navigation step does not have a goal pose available.",
                "goal_pose_missing",
                started_at,
            )

        try:
            return self._ros_client.navigate_to_pose(
                step_id=step.step_id,
                goal_pose=goal_pose,
                execution_backend=str(
                    step.args.get("execution_backend", self._config.default_navigation_backend)
                ),
                behavior_tree=str(step.args["behavior_tree"]) if step.args.get("behavior_tree") else None,
                timeout_s=step.timeout_s or self._config.default_navigation_timeout_s,
            )
        except Exception as exc:
            return self._failed_result(step, f"Navigation request failed: {exc}", "navigation_failed", started_at)

    def _orient_relative_to_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        candidate = runtime.latest_goal_pose
        target_pose = candidate.target_pose if candidate is not None else None
        if target_pose is None:
            target_pose = self._parse_pose3d(step.args.get("target_pose"))
        if target_pose is None:
            return self._failed_result(
                step,
                "Orientation step does not have a target pose available.",
                "target_pose_missing",
                started_at,
            )

        mode = str(step.args.get("mode", "")).strip()
        if not mode:
            return self._failed_result(step, "Orientation step is missing 'mode'.", "invalid_args", started_at)

        try:
            return self._ros_client.orient_relative_to_target(
                step_id=step.step_id,
                target_pose=target_pose,
                mode=mode,
                yaw_offset_rad=float(step.args.get("yaw_offset_rad", 0.0)),
                tolerance_rad=float(step.args.get("tolerance_rad", 0.1)),
                timeout_s=step.timeout_s,
            )
        except Exception as exc:
            return self._failed_result(step, f"Orientation request failed: {exc}", "orientation_failed", started_at)

    def _wait(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        mode = str(step.args.get("mode", "")).strip() or "duration"
        duration_s = step.args.get("duration_s")
        deadline = None
        if mode != "until_next_command":
            if duration_s is None:
                duration_s = step.timeout_s
            if duration_s is not None:
                deadline = time.monotonic() + float(duration_s)

        while True:
            if runtime.cancel_event.is_set():
                return SkillResult(
                    step_id=step.step_id,
                    skill=step.skill,
                    status="canceled",
                    outputs={"mode": mode},
                    error_code="mission_canceled",
                    message="Wait step canceled.",
                    started_at=started_at,
                    finished_at=time.time(),
                )
            if deadline is not None and time.monotonic() >= deadline:
                return SkillResult(
                    step_id=step.step_id,
                    skill=step.skill,
                    status="succeeded",
                    outputs={"mode": mode, "duration_s": float(duration_s)},
                    message="Wait step completed.",
                    started_at=started_at,
                    finished_at=time.time(),
                )
            if mode == "until_next_command":
                if runtime.resume_event.is_set():
                    return SkillResult(
                        step_id=step.step_id,
                        skill=step.skill,
                        status="succeeded",
                        outputs={"mode": mode, "resumed": True},
                        message="Wait step resumed by new command.",
                        started_at=started_at,
                        finished_at=time.time(),
                    )
                time.sleep(self._config.wait_poll_period_s)
                continue
            if deadline is None:
                return self._failed_result(
                    step,
                    "Wait step requires either mode='until_next_command' or a duration/timeout.",
                    "invalid_args",
                    started_at,
                )
            time.sleep(self._config.wait_poll_period_s)

    def _report_status(self, step: SkillCall) -> SkillResult:
        started_at = time.time()
        try:
            robot_state = self._ros_client.get_robot_state()
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs={"robot_state": robot_state},
                message="Reported robot state.",
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            return self._failed_result(step, f"Robot status query failed: {exc}", "robot_state_failed", started_at)

    def _cancel_step(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
        started_at = time.time()
        self.cancel_active_mission()
        return SkillResult(
            step_id=step.step_id,
            skill=step.skill,
            status="canceled",
            outputs={},
            error_code="mission_canceled",
            message="Mission canceled by plan step.",
            started_at=started_at,
            finished_at=time.time(),
        )

    def _safe_get_robot_state(self) -> dict[str, Any] | None:
        try:
            return self._ros_client.get_robot_state()
        except Exception:
            return None

    def _set_runtime_state(
        self,
        runtime: _MissionRuntime,
        *,
        state: str,
        message: str,
        current_step_id: str | None = None,
        current_skill: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            runtime.state = state
            runtime.message = message
            if current_step_id is not None:
                runtime.current_step_id = current_step_id
            if current_skill is not None:
                runtime.current_skill = current_skill
            if error_code is not None:
                runtime.error_code = error_code

    def _finish_runtime(
        self,
        runtime: _MissionRuntime,
        *,
        state: str,
        message: str,
        error_code: str,
    ) -> None:
        with self._lock:
            runtime.active = False
            runtime.state = state
            runtime.message = message
            runtime.error_code = error_code
            if self._active_runtime is runtime:
                self._active_runtime = None
            self._last_runtime = runtime

    def _request_cancel_locked(
        self,
        runtime: _MissionRuntime,
        *,
        state: str,
        message: str,
        error_code: str,
    ) -> None:
        runtime.cancel_event.set()
        runtime.state = state
        runtime.message = message
        runtime.error_code = error_code

    def _snapshot_from_runtime_locked(self, runtime: _MissionRuntime) -> MissionStatusSnapshot:
        elapsed_s = max(0.0, time.time() - runtime.started_at)
        return MissionStatusSnapshot(
            active=runtime.active,
            mission_id=runtime.mission_id,
            state=runtime.state,
            raw_command=runtime.raw_command,
            current_step_id=runtime.current_step_id,
            current_skill=runtime.current_skill,
            message=runtime.message,
            error_code=runtime.error_code,
            elapsed_s=elapsed_s,
        )

    def _active_mission_summary(self, runtime: _MissionRuntime) -> dict[str, Any] | None:
        with self._lock:
            if self._last_runtime is None:
                return None
            last = self._last_runtime
            return {
                "mission_id": last.mission_id,
                "state": last.state,
                "current_step_id": last.current_step_id,
                "current_skill": last.current_skill,
                "message": last.message,
                "same_request": last.request_id == runtime.request_id,
            }

    def _failed_result(
        self,
        step: SkillCall,
        message: str,
        error_code: str,
        started_at: float | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(
            step_id=step.step_id,
            skill=step.skill,
            status="failed",
            outputs=outputs or {},
            error_code=error_code,
            message=message,
            started_at=started_at or time.time(),
            finished_at=time.time(),
        )

    def _cancel_robot_actions(self) -> None:
        try:
            self._ros_client.cancel_active_navigation()
        except Exception:
            return

    def _bgr_to_rgb(self, image: Any) -> Any:
        try:
            return image[..., ::-1]
        except Exception:
            return image

    def _grounding_outputs(self, grounding: GroundingResult) -> dict[str, Any]:
        return {
            "request_id": grounding.request_id,
            "found": grounding.found,
            "bbox_2d": list(grounding.bbox_2d) if grounding.bbox_2d is not None else None,
            "label": grounding.label,
            "confidence": grounding.confidence,
            "latency_s": grounding.latency_s,
            "debug_overlay_jpeg_b64": grounding.debug_overlay_jpeg_b64,
        }

    def _goal_pose_outputs(self, candidate: GoalPoseCandidate) -> dict[str, Any]:
        return {
            "request_id": candidate.request_id,
            "found": candidate.found,
            "goal_frame": candidate.goal_frame,
            "goal_pose": self._serialize(candidate.goal_pose),
            "target_pose": self._serialize(candidate.target_pose),
            "standoff_m": candidate.standoff_m,
            "depth_valid": candidate.depth_valid,
            "quality_flags": list(candidate.quality_flags),
            "message": candidate.message,
        }

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, tuple):
            return [self._serialize(item) for item in value]
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._serialize(item) for key, item in value.items()}
        return value

    @staticmethod
    def _parse_pose3d(raw: Any) -> Pose3D | None:
        """Try to convert a raw dict (e.g. from planner args) into a ``Pose3D``."""
        if isinstance(raw, Pose3D):
            return raw
        if not isinstance(raw, dict):
            return None
        try:
            return Pose3D(
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                z=float(raw.get("z", 0.0)),
                qx=float(raw.get("qx", 0.0)),
                qy=float(raw.get("qy", 0.0)),
                qz=float(raw.get("qz", 0.0)),
                qw=float(raw.get("qw", 1.0)),
            )
        except (TypeError, ValueError):
            return None
