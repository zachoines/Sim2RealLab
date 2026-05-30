"""End-to-end round-trip tests for the StraferLeRobotWriter.

Exercises the full lifecycle: ``create()`` → ``add_frame()`` →
``save_episode()`` → ``finalize()`` from the harness writer's wrapper,
then re-loads the dataset via stock LeRobotDataset and confirms the
schema, frame counts, and strafer sidecars all round-trip.

Runs in ``.venv_harness`` only (requires lerobot, not Isaac Sim).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from strafer_lab.tools.lerobot_depth import (
    POLICY_DEPTH,
    frame_path,
    read_depth_png,
)
from strafer_lab.tools.lerobot_writer import (
    StraferLeRobotWriter,
    _DepthWriterPool,
    build_features,
    read_strafer_episodes,
)


@pytest.fixture()
def writer_root():
    """Yield a non-existent dataset root inside a temp tree."""
    parent = Path(tempfile.mkdtemp(prefix="strafer_writer_test_"))
    yield parent / "dataset"
    shutil.rmtree(parent, ignore_errors=True)


def _make_rgb(h: int = 360, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)


def _make_depth(h: int = 360, w: int = 640, base_m: float = 1.5) -> np.ndarray:
    return np.full((h, w), base_m, dtype=np.float32)


# ---------------------------------------------------------------------------
# build_features schema
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_includes_perception_video_state_action(self):
        feats = build_features(capture_policy_cam=False)
        assert "observation.images.perception" in feats
        assert "observation.state" in feats
        assert "action" in feats
        assert feats["observation.images.perception"]["dtype"] == "video"
        assert feats["observation.images.perception"]["shape"] == (360, 640, 3)

    def test_policy_cam_optional(self):
        no_policy = build_features(capture_policy_cam=False)
        with_policy = build_features(capture_policy_cam=True)
        assert "observation.images.policy" not in no_policy
        assert "observation.images.policy" in with_policy
        assert with_policy["observation.images.policy"]["shape"] == (60, 80, 3)

    def test_state_action_dims(self):
        feats = build_features()
        assert feats["observation.state"]["shape"] == (10,)
        assert feats["action"]["shape"] == (3,)
        assert "pose_x" in feats["observation.state"]["names"]
        assert feats["action"]["names"] == ["vx_cmd", "vy_cmd", "omega_z_cmd"]


# ---------------------------------------------------------------------------
# Writer end-to-end round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_single_episode_with_depth(self, writer_root):
        """Write 5 frames + depth, finalize, re-load via LeRobotDataset."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/single-episode",
            fps=8,
            capture_git_sha="deadbeef",
            scene_metadata_hash="hash123",
            capture_policy_cam=True,
            operator_handle="z",
            session_id="20260524_120000",
        ) as writer:
            ep_idx = writer.begin_episode(
                mission_text="go to the chair",
                scene_id="scene_001",
                target_label="chair",
                target_object_id="chair_3",
                target_position_3d=[4.2, 1.8, 0.0],
                start_pose=[0.5, 0.5, 0.0],
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            assert ep_idx == 0

            for t in range(5):
                writer.add_frame(
                    sim_time=float(t) / 8.0,
                    pose=[float(t), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    achieved_vel=[1.0, 0.0, 0.0],
                    action=[1.0, 0.0, 0.0],
                    rgb_perception=_make_rgb(360, 640),
                    rgb_policy=_make_rgb(60, 80),
                    depth_m=_make_depth(360, 640, base_m=2.0 + t * 0.1),
                )

            writer.end_episode(
                outcome="succeeded",
                outcome_category="on_course",
            )

        # --- read back via stock LeRobotDataset ---
        reloaded = LeRobotDataset(
            repo_id="strafer-test/single-episode",
            root=writer_root,
        )
        assert len(reloaded) == 5
        assert reloaded.num_episodes == 1

        # Sample shape/typing
        sample = reloaded[0]
        assert "observation.images.perception" in sample
        assert "observation.images.policy" in sample
        assert "observation.state" in sample
        assert "action" in sample
        # Images are CHW tensors by LeRobot convention
        assert tuple(sample["observation.images.perception"].shape)[-2:] == (640,) * 0 or \
               sample["observation.images.perception"].shape[1:] == (360, 640) or \
               sample["observation.images.perception"].shape[:2] == (360, 640)

    def test_strafer_sidecar_round_trips(self, writer_root):
        """Per-episode strafer extension columns survive the round-trip."""
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/sidecar",
            fps=8,
            capture_git_sha="abc123",
            scene_metadata_hash="sha256xyz",
            capture_policy_cam=False,
            operator_handle="alice",
            session_id="sess_001",
        ) as writer:
            writer.begin_episode(
                mission_text="approach the bed",
                scene_id="scene_bedroom",
                target_label="bed",
                source_driver="teleop",
                source_mission_source="scene-metadata",
                paraphrases=["go to the bed", "head to the bed"],
            )
            for t in range(3):
                writer.add_frame(
                    sim_time=float(t) / 8.0,
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.5, 0.0, 0.0],
                    rgb_perception=_make_rgb(360, 640),
                )
            writer.end_episode(outcome="succeeded", outcome_category="on_course")

        episodes = read_strafer_episodes(writer_root)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["scene_id"] == "scene_bedroom"
        assert ep["target_label"] == "bed"
        assert ep["outcome"] == "succeeded"
        assert ep["outcome_category"] == "on_course"
        assert ep["source_driver"] == "teleop"
        assert ep["paraphrases"] == ["go to the bed", "head to the bed"]
        assert ep["capture_git_sha"] == "abc123"
        assert ep["scene_metadata_hash"] == "sha256xyz"
        assert ep["operator_handle"] == "alice"
        assert ep["session_id"] == "sess_001"
        assert ep["episode_split"] == "val"  # episode_index 0 → val per the % 10 rule

    def test_depth_sidecar_persists(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/depth",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            for t in range(3):
                writer.add_frame(
                    sim_time=float(t),
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(),
                    depth_m=_make_depth(base_m=1.0 + t * 0.5),
                )
            writer.end_episode()

        # All 3 frames have a depth PNG under the deterministic path
        for f in range(3):
            png = frame_path(writer_root, 0, f)
            assert png.is_file(), f"missing depth PNG for frame {f}: {png}"
            recovered = read_depth_png(png)
            expected = 1.0 + f * 0.5
            assert np.allclose(recovered, expected, atol=1e-3)

    def test_discard_does_not_advance_episode(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/discard",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        ) as writer:
            # Episode 0: discarded
            ep0 = writer.begin_episode(
                mission_text="discard-me",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            for t in range(2):
                writer.add_frame(
                    sim_time=float(t),
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(),
                )
            writer.end_episode(discard=True)
            assert ep0 == 0
            assert writer.num_episodes == 0

            # Episode 0 (reused index): kept
            ep1 = writer.begin_episode(
                mission_text="keep-me",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            assert ep1 == 0  # same index reused
            for t in range(2):
                writer.add_frame(
                    sim_time=float(t),
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(),
                )
            writer.end_episode(outcome="succeeded", outcome_category="on_course")
            assert writer.num_episodes == 1

        # Only one episode persisted; sidecar reflects the kept one.
        episodes = read_strafer_episodes(writer_root)
        assert len(episodes) == 1
        assert episodes[0]["episode_index"] == 0


# ---------------------------------------------------------------------------
# Async depth writer
# ---------------------------------------------------------------------------


class TestAsyncDepthWriting:
    """End-to-end behaviour of the depth-sidecar worker pool through the writer."""

    def test_synchronous_depth_path_still_works(self, writer_root):
        """``depth_writer_threads=0`` should fall back to the inline encode.

        The fully-synchronous path is the debugging mode (e.g., when
        chasing a frame-ordering bug) — exercise it so it doesn't bit-rot.
        """
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/depth-sync",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
            depth_writer_threads=0,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            for t in range(3):
                writer.add_frame(
                    sim_time=float(t),
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(),
                    depth_m=_make_depth(base_m=0.75 + t * 0.25),
                )
            writer.end_episode()

        for f in range(3):
            png = frame_path(writer_root, 0, f)
            assert png.is_file()
            recovered = read_depth_png(png)
            assert np.allclose(recovered, 0.75 + f * 0.25, atol=1e-3)

    def test_async_depth_pool_captures_post_mutation_snapshot(self, writer_root):
        """Submitting a depth array then mutating it must not corrupt the PNG.

        The worker copies on submit precisely so the env-step thread can
        reuse Isaac Sim's render buffer for the next frame. This test
        proves the copy is happening.
        """
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/depth-mutate",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
            depth_writer_threads=2,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            shared_buf = _make_depth(base_m=1.25)
            writer.add_frame(
                sim_time=0.0,
                pose=[0.0] * 7,
                achieved_vel=[0.0] * 3,
                action=[0.0] * 3,
                rgb_perception=_make_rgb(),
                depth_m=shared_buf,
            )
            # Mimic Isaac Sim recycling the render buffer before the
            # worker thread gets scheduled.
            shared_buf.fill(99.0)
            writer.end_episode()

        recovered = read_depth_png(frame_path(writer_root, 0, 0))
        assert np.allclose(recovered, 1.25, atol=1e-3)


class TestPolicyDepthSidecar:
    """Policy-camera depth rides as its own feature-named sidecar."""

    def test_policy_depth_lands_under_policy_feature_dir(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/policy-depth",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=True,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            for t in range(2):
                writer.add_frame(
                    sim_time=float(t),
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(360, 640),
                    rgb_policy=_make_rgb(60, 80),
                    depth_m=_make_depth(360, 640, base_m=2.0),
                    depth_policy_m=_make_depth(60, 80, base_m=1.25),
                )
            writer.end_episode()

        for f in range(2):
            policy_png = frame_path(writer_root, 0, f, POLICY_DEPTH)
            perception_png = frame_path(writer_root, 0, f)  # default = perception
            assert policy_png.is_file(), f"missing policy depth PNG: {policy_png}"
            assert perception_png.is_file()
            # The two feature dirs are distinct so frames don't collide.
            assert policy_png.parent != perception_png.parent
            assert np.allclose(read_depth_png(policy_png), 1.25, atol=1e-3)
            assert np.allclose(read_depth_png(perception_png), 2.0, atol=1e-3)

    def test_policy_depth_is_optional(self, writer_root):
        """Without depth_policy_m, no policy-depth dir is created."""
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="strafer-test/policy-depth-skip",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=True,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            writer.add_frame(
                sim_time=0.0,
                pose=[0.0] * 7,
                achieved_vel=[0.0] * 3,
                action=[0.0] * 3,
                rgb_perception=_make_rgb(360, 640),
                rgb_policy=_make_rgb(60, 80),
                depth_m=_make_depth(360, 640),
            )
            writer.end_episode()

        assert not (writer_root / "videos" / POLICY_DEPTH).exists()


class TestDepthWriterPool:
    """Unit tests for the depth worker pool in isolation."""

    def test_worker_exception_resurfaces_on_drain(self, tmp_path):
        """A bad disk write should not be silently swallowed."""
        pool = _DepthWriterPool(num_workers=1)
        try:
            bad_path = tmp_path / "ok.png"
            bad_array = np.zeros((4, 4, 3), dtype=np.float32)  # rank 3 → invalid
            pool.submit(bad_path, bad_array)
            with pytest.raises(ValueError, match="depth_m must be"):
                pool.wait_until_done()
        finally:
            pool.stop()

    def test_drain_clears_error_state(self, tmp_path):
        """After a drain re-raises an exception, the next drain is clean."""
        pool = _DepthWriterPool(num_workers=1)
        try:
            pool.submit(tmp_path / "bad.png", np.zeros((4, 4, 3), dtype=np.float32))
            with pytest.raises(ValueError):
                pool.wait_until_done()
            good = np.full((4, 4), 1.0, dtype=np.float32)
            pool.submit(tmp_path / "good.png", good)
            pool.wait_until_done()
            assert (tmp_path / "good.png").is_file()
        finally:
            pool.stop()

    def test_stop_is_idempotent(self):
        pool = _DepthWriterPool(num_workers=2)
        pool.stop()
        pool.stop()  # second stop should be a no-op, not block forever


# ---------------------------------------------------------------------------
# Lifecycle guards
# ---------------------------------------------------------------------------


class TestLifecycleGuards:
    def test_double_begin_raises(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="r",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            with pytest.raises(RuntimeError, match="another episode is open"):
                writer.begin_episode(
                    mission_text="t2",
                    scene_id="s",
                    source_driver="teleop",
                    source_mission_source="scene-metadata",
                )
            writer.end_episode(discard=True)

    def test_add_frame_without_episode_raises(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="r",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        ) as writer:
            with pytest.raises(RuntimeError, match="outside of an open episode"):
                writer.add_frame(
                    sim_time=0.0,
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(),
                )

    def test_finalize_with_open_episode_discards(self, writer_root):
        """An open episode at finalize time discards rather than persists."""
        writer = StraferLeRobotWriter(
            root=writer_root,
            repo_id="r",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        )
        writer.begin_episode(
            mission_text="t",
            scene_id="s",
            source_driver="teleop",
            source_mission_source="scene-metadata",
        )
        # Frame written but never end_episode()ed
        writer.add_frame(
            sim_time=0.0,
            pose=[0.0] * 7,
            achieved_vel=[0.0] * 3,
            action=[0.0] * 3,
            rgb_perception=_make_rgb(),
        )
        writer.finalize()
        # No persisted episode
        assert writer.num_episodes == 0
        # finalize is idempotent
        writer.finalize()

    def test_rejects_wrong_rgb_shape(self, writer_root):
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id="r",
            fps=8,
            capture_git_sha="x",
            scene_metadata_hash="y",
            capture_policy_cam=False,
        ) as writer:
            writer.begin_episode(
                mission_text="t",
                scene_id="s",
                source_driver="teleop",
                source_mission_source="scene-metadata",
            )
            with pytest.raises(ValueError, match=r"shape"):
                writer.add_frame(
                    sim_time=0.0,
                    pose=[0.0] * 7,
                    achieved_vel=[0.0] * 3,
                    action=[0.0] * 3,
                    rgb_perception=_make_rgb(h=100, w=100),  # wrong shape
                )
            writer.end_episode(discard=True)
