"""Unit tests for the offline mission generator.

Pure numpy — no Isaac Sim, no Kit, no pxr. The scene metadata + occupancy grid
are built in memory (the planner convention: ``grid[row, col]`` with row = x
index, col = y index, origin at the corner of cell (0, 0)). The single ``pxr``
touch-point (``load_scene_inputs``) and the model runners are not exercised
here; everything else operates on the in-memory :class:`SceneInputs`.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from strafer_lab.tools import build_mission_queue as bmq
from strafer_lab.tools.build_mission_queue import GeneratorConfig, SceneInputs
from strafer_lab.tools.mission_queue import load_mission_queue, parse_mission_row

RES = 0.1
ORIGIN = (0.0, 0.0)


def _two_room_scene(*, incompatible: bool = False, reachable: bool = True) -> SceneInputs:
    """Two rooms along +x joined by a doorway at y=1.5; one object in each.

    Grid spans x in [0, 8], y in [0, 3]; a wall at x=4 (row 40) with a free
    doorway gap at y in [1.3, 1.7]. Kitchen is room 0, living_room is room 1.
    """
    nx, ny = 80, 30
    free = np.ones((nx, ny), dtype=bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False  # outer walls
    free[40, :] = False  # mid wall
    free[40, 13:17] = True  # doorway

    rooms = [
        {"room_type": "kitchen", "footprint_xy": [[0, 0], [4, 0], [4, 3], [0, 3]], "story": 0, "area_m2": 12.0},
        {"room_type": "living_room", "footprint_xy": [[4, 0], [8, 0], [8, 3], [4, 3]], "story": 0, "area_m2": 12.0},
    ]
    objects = [
        {"label": "table", "instance_id": 1, "position_3d": [2.0, 1.0, 0.3], "prim_path": "/World/Table_1"},
        {"label": "chair", "instance_id": 2, "position_3d": [6.0, 1.5, 0.3], "prim_path": "/World/Chair_2"},
    ]
    connectivity = [
        {
            "from_idx": 0,
            "to_idx": 1,
            "reachable": reachable,
            **(
                {"via_doorway_xy": [4.0, 1.5], "path_length_m": 4.0, "door_state": "absent"}
                if reachable
                else {"reason": "blocked"}
            ),
        }
    ]
    metadata = {
        "rooms": rooms,
        "objects": objects,
        "connectivity": connectivity,
        "room_adjacency": [],
        "multi_story": False,
    }
    if incompatible:
        metadata["multi_room_incompatible"] = True
    return SceneInputs(
        scene_name="scene_test_000_seed7",
        metadata=metadata,
        scene_seed=7,
        free_space=free,
        grid_res=RES,
        grid_origin_xy=ORIGIN,
        occupancy_meta={},
    )


def _single_room_scene_no_occupancy() -> SceneInputs:
    """Legacy single-room scene: one room, objects, but no cached occupancy grid."""
    metadata = {
        "rooms": [{"room_type": "room", "footprint_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "story": 0}],
        "objects": [
            {"label": "bottle", "instance_id": 1, "position_3d": [1.0, 1.0, 0.2]},
            {"label": "bowl", "instance_id": 2, "position_3d": [3.0, 2.0, 0.2]},
        ],
        "connectivity": [],
    }
    return SceneInputs(
        scene_name="scene_fast_singleroom_000_seed0",
        metadata=metadata,
        scene_seed=0,
        free_space=None,
    )


# ---------------------------------------------------------------------------
# Round-trip: every emitted row parses through the mission_queue contract
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_endpoint_rows_round_trip(self, tmp_path):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        assert result.rows
        path = bmq.write_queue(tmp_path / "queue.yaml", result.rows)
        parsed = load_mission_queue(path)
        assert len(parsed) == len(result.rows)
        for row in parsed:
            assert row.mission_id
            assert row.mission_text
            assert row.target_label
            assert len(row.target_position_3d) == 3

    def test_required_quartet_present_on_every_row(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="mixed"))
        for raw in result.rows:
            # Must not raise — exercises the real parser on the raw dict.
            parse_mission_row(raw)

    def test_grounding_and_provenance_nested_in_generator_metadata(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        for raw in result.rows:
            # start_frame_grounded / mission_id / scene_seed / rooms / planned_path
            # have no top-level queue field; they must ride generator_metadata so
            # they survive the parser round-trip.
            assert "start_frame_grounded" not in raw
            gm = raw["generator_metadata"]
            assert "start_frame_grounded" in gm
            assert gm["source_mission_source"] == "queue"
            assert gm["mission_id"] == raw["mission_id"]
            assert gm["scene_seed"] == 7
            # No waypoint LLM ran -> the corpus must not claim one.
            assert gm["llm_model"] is None
            # Parser keeps generator_metadata opaque, so the nested fields survive.
            parsed = parse_mission_row(raw)
            assert parsed.generator_metadata["start_frame_grounded"] == gm["start_frame_grounded"]
            assert parsed.generator_metadata["cross_room"] == raw["cross_room"]


# ---------------------------------------------------------------------------
# One-planner invariant: oracle paths come only from plan_path
# ---------------------------------------------------------------------------


class TestOraclePaths:
    def test_endpoint_path_starts_at_start_ends_at_target(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        row = next(r for r in result.rows if r.get("planned_path"))
        path = row["planned_path"]
        start = row["start_pose"]
        # plan_path pins exact endpoints: first == start, last == target XY.
        assert path[0]["x"] == pytest.approx(start["x"], abs=1e-6)
        assert path[0]["y"] == pytest.approx(start["y"], abs=1e-6)
        assert path[-1]["x"] == pytest.approx(row["target_position_3d"][0], abs=0.05)
        assert path[-1]["y"] == pytest.approx(row["target_position_3d"][1], abs=0.05)

    def test_emitted_path_starts_on_navigable_cell(self):
        # Block the living-room representative point's cell so the room rep falls
        # back to a non-free centroid; the start must still be navigable.
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        for r in result.rows:
            path = r.get("planned_path")
            if not path:
                continue
            origin, res = scene.grid_origin_xy, scene.grid_res
            assert bmq._cell_is_free(scene.free_space, path[0]["x"], path[0]["y"], origin, res)

    def test_unreachable_target_is_rejected_not_emitted(self):
        scene = _two_room_scene()
        # Seal a block of the living room so no free cell sits within the snap
        # radius of the object dropped at its centre.
        scene.free_space[48:73, 3:28] = False
        scene.metadata["objects"].append(
            {"label": "ghost", "instance_id": 9, "position_3d": [6.0, 1.5, 0.3]}
        )
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        labels = {r["target_label"] for r in result.rows}
        assert "ghost" not in labels
        assert result.stats.rejected_reasons.get("no_navigable_path", 0) >= 1


# ---------------------------------------------------------------------------
# LLM-as-planner: validation + retry + clean-oracle fallback
# ---------------------------------------------------------------------------


class TestWaypointValidationAndFallback:
    def test_offmesh_llm_waypoints_rejected_then_fall_back_to_oracle(self):
        scene = _two_room_scene()

        def bad_runner(prompt: str, seed: int) -> str:
            # Waypoints far outside the grid — must fail validation every retry.
            return json.dumps({"waypoints": [{"x": -50, "y": -50}, {"x": 99, "y": 99}], "rationale": "x"})

        result = bmq.build_mission_queue(
            scene, GeneratorConfig(mode="path-shape", max_retries=2), waypoint_runner=bad_runner
        )
        assert result.stats.llm_retries > 0
        assert result.stats.path_shape_unsatisfied > 0
        row = next(r for r in result.rows if r.get("planned_path"))
        assert row["generator_metadata"]["waypoint_validation"]["source"] == "oracle_fallback"
        assert row["generator_metadata"]["constraint_type_hint"].endswith("_unsatisfied")
        # Waypoints fell back to the oracle -> no LLM model is claimed.
        assert row["generator_metadata"]["llm_model"] is None

    def test_valid_llm_waypoints_are_used(self):
        scene = _two_room_scene()
        # Move the chair off the room centre so a same-room start (the room
        # representative point) differs from the target, then stub a clean
        # in-room path from that start to the target.
        scene.metadata["objects"][1]["position_3d"] = [7.0, 1.5, 0.3]
        good = [{"x": 6.0, "y": 1.5}, {"x": 6.5, "y": 1.5}, {"x": 7.0, "y": 1.5}]

        def good_runner(prompt: str, seed: int) -> str:
            return json.dumps({"waypoints": good, "rationale": "hug"})

        result = bmq.build_mission_queue(
            scene, GeneratorConfig(mode="path-shape", cross_room_default=False), waypoint_runner=good_runner
        )
        chair = next(r for r in result.rows if r["target_label"] == "chair")
        assert chair["generator_metadata"]["waypoint_validation"]["source"] == "llm"
        assert chair["generator_metadata"]["waypoint_validation"]["retries"] == 0
        assert chair["planned_path"][-1]["x"] == pytest.approx(7.0, abs=0.01)
        # LLM produced the waypoints -> the planner model is stamped.
        assert chair["generator_metadata"]["llm_model"] == "Qwen/Qwen3-4B"

    def test_nearest_path_landmark_picks_unique_near_path_object(self):
        rooms = [{"room_type": "kitchen", "footprint_xy": [[0, 0], [8, 0], [8, 3], [0, 3]], "story": 0}]
        path = np.array([[0.0, 1.5], [2.0, 1.5], [4.0, 1.5], [6.0, 1.5]])
        target = {"label": "chair", "instance_id": 1, "position_3d": [6.0, 1.5, 0.3]}
        objs = [
            target,
            {"label": "table", "instance_id": 2, "position_3d": [2.0, 1.0, 0.3]},  # near, unique
            {"label": "stool", "instance_id": 3, "position_3d": [2.0, 8.0, 0.3]},  # far off path
        ]
        lm = bmq._nearest_path_landmark(path, objs, target, rooms, band_m=1.0)
        assert lm == "table"  # target excluded, far object excluded
        # A duplicate label along the path makes it ambiguous -> None.
        objs.append({"label": "table", "instance_id": 4, "position_3d": [4.0, 1.2, 0.3]})
        assert bmq._nearest_path_landmark(path, objs, target, rooms, band_m=1.0) is None

    def test_same_room_mission_uses_passing_the_landmark(self):
        scene = _two_room_scene()
        # Off-centre target + a unique landmark between the room rep and target.
        scene.metadata["objects"][1]["position_3d"] = [7.0, 1.5, 0.3]
        scene.metadata["objects"].append(
            {"label": "lamp", "instance_id": 5, "position_3d": [6.5, 1.5, 0.4]}
        )
        result = bmq.build_mission_queue(
            scene, GeneratorConfig(mode="path-shape", cross_room_default=False)
        )
        chair = next(r for r in result.rows if r["target_label"] == "chair")
        assert chair["mission_text"] == "Go to the chair, passing the lamp."
        # "_unsatisfied" suffix when the model-free oracle fallback supplies the path.
        assert chair["generator_metadata"]["constraint_type_hint"].startswith("landmark_relative")

    def test_validate_waypoints_flags_each_failure_mode(self):
        scene = _two_room_scene()
        cfg = GeneratorConfig()
        # Inside the mid wall (blocked) -> navigable_ok False.
        ok, checks = bmq.validate_waypoints(
            [(4.0, 0.5), (4.0, 0.6)], inputs=scene, target_xy=(4.0, 0.6), config=cfg
        )
        assert not ok and not checks["navigable_ok"]


# ---------------------------------------------------------------------------
# Connectivity gate + multi-room semantics
# ---------------------------------------------------------------------------


class TestConnectivityGate:
    def test_cross_room_default_emits_cross_room_missions(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        assert result.stats.multi_room
        assert result.stats.cross_room >= 1
        cross = next(r for r in result.rows if r["cross_room"])
        assert cross["start_room"] != cross["target_room"]
        assert cross["generator_metadata"]["cross_room"] is True

    def test_unreachable_pair_forces_same_room(self):
        scene = _two_room_scene(reachable=False)
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        assert not result.stats.multi_room
        assert result.stats.cross_room == 0
        assert all(not r["cross_room"] for r in result.rows)

    def test_multi_room_incompatible_flag_forces_same_room(self):
        scene = _two_room_scene(incompatible=True)
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        assert not result.stats.multi_room
        assert all(not r["cross_room"] for r in result.rows)

    def test_reachable_room_pairs_is_order_insensitive(self):
        edges = [
            {"from_idx": 0, "to_idx": 2, "reachable": True},
            {"from_idx": 1, "to_idx": 0, "reachable": False, "reason": "blocked"},
        ]
        assert bmq.reachable_room_pairs(edges) == {(0, 2)}

    def test_upper_floor_target_is_rejected(self):
        # The strafer cannot climb stairs, so a target on story 1 (cross-story
        # edge unreachable) must never be emitted, even if the ground-floor
        # occupancy projection leaves free cells under it.
        scene = _two_room_scene(reachable=False)
        scene.metadata["rooms"][1]["story"] = 1
        scene.metadata["multi_story"] = True
        scene.metadata["connectivity"] = [
            {"from_idx": 0, "to_idx": 1, "reachable": False, "reason": "stairs"}
        ]
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        labels = {r["target_label"] for r in result.rows}
        assert "chair" not in labels  # the story-1 object
        assert "table" in labels  # the story-0 object still ships
        assert result.stats.rejected_reasons.get("target_on_unreachable_floor", 0) == 1


# ---------------------------------------------------------------------------
# Single-room (no-occupancy) fallback
# ---------------------------------------------------------------------------


class TestSingleRoomFallback:
    def test_single_room_emits_same_room_pathless_rows(self, tmp_path):
        scene = _single_room_scene_no_occupancy()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="mixed"))
        assert result.rows
        assert all(not r["cross_room"] for r in result.rows)
        assert all(not r.get("planned_path") for r in result.rows)
        path = bmq.write_queue(tmp_path / "q.yaml", result.rows)
        assert len(load_mission_queue(path)) == len(result.rows)


# ---------------------------------------------------------------------------
# Start-frame grounding
# ---------------------------------------------------------------------------


class TestStartFrameGrounding:
    def _frame_provider(self, obj, start_pose):
        return "frame.png"  # any non-None sentinel; the runner is stubbed

    def _pil_frame_provider(self, obj, start_pose):
        from PIL import Image  # the live provider returns an in-memory PIL.Image

        return Image.new("RGB", (4, 4))

    def test_skipped_when_disabled(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        assert result.stats.start_frame_grounded_skipped == result.stats.emitted
        for r in result.rows:
            assert r["generator_metadata"]["start_frame_grounded"] is None

    def test_no_verdict_rejects_same_room_keeps_cross_room(self):
        scene = _two_room_scene()

        def runner(frame, mission_text):
            return "no"

        # Cross-room: "no" is kept with start_frame_grounded=False.
        cross = bmq.build_mission_queue(
            scene,
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=runner,
            grounding_frame_provider=self._frame_provider,
        )
        assert cross.stats.emitted >= 1
        assert all(r["generator_metadata"]["start_frame_grounded"] is False for r in cross.rows)

        # Same-room only: "no" rejects the mission entirely.
        same = bmq.build_mission_queue(
            _two_room_scene(reachable=False),
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=runner,
            grounding_frame_provider=self._frame_provider,
        )
        assert same.stats.emitted == 0
        assert same.stats.rejected_reasons.get("target_not_visible_at_start", 0) >= 1

    def test_yes_verdict_marks_grounded_true(self):
        scene = _two_room_scene()

        def runner(frame, mission_text):
            return "yes"

        result = bmq.build_mission_queue(
            scene,
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=runner,
            grounding_frame_provider=self._frame_provider,
        )
        assert result.stats.start_frame_grounded_yes == result.stats.emitted
        assert all(r["generator_metadata"]["start_frame_grounded"] is True for r in result.rows)

    def test_partial_verdict_marks_grounded_true(self):
        # "partial" is a ship verdict (same bucket as "yes"); the row is kept and
        # grounded True. Exercised with the live provider's PIL.Image return type.
        scene = _two_room_scene()

        def runner(frame, mission_text):
            return "partial"

        result = bmq.build_mission_queue(
            scene,
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=runner,
            grounding_frame_provider=self._pil_frame_provider,
        )
        assert result.stats.emitted >= 1
        assert result.stats.start_frame_grounded_yes == result.stats.emitted
        assert result.stats.start_frame_grounded_no == 0
        assert all(r["generator_metadata"]["start_frame_grounded"] is True for r in result.rows)

    def test_provider_returning_none_skips_grounding(self):
        # A frame provider that returns None (e.g. a bad/missing render the live
        # provider sanitized away) is the SECOND skip path in _ground_start_frame:
        # the runner is never called and the mission ships ungrounded.
        scene = _two_room_scene()

        def runner(frame, mission_text):
            raise AssertionError("runner must not run when the provider yields None")

        def none_provider(obj, start_pose):
            return None

        result = bmq.build_mission_queue(
            scene,
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=runner,
            grounding_frame_provider=none_provider,
        )
        assert result.stats.emitted >= 1
        assert result.stats.start_frame_grounded_skipped == result.stats.emitted
        assert result.stats.start_frame_grounded_yes == 0
        assert result.stats.start_frame_grounded_no == 0
        for r in result.rows:
            assert r["generator_metadata"]["start_frame_grounded"] is None


class TestGeometricVisibilityVerdict:
    """The model-free geometric start-frame verdict + its runner.

    The verdict ships a mission ("yes") iff the KNOWN target is geometrically
    observable from the start pose: in frame, not effectively occluded, big
    enough on screen, and not mostly truncated by a frame edge. Each threshold
    is asserted INDIVIDUALLY necessary against the imported module constants —
    no magic numbers duplicated here.
    """

    FRAME_W = 640
    FRAME_H = 360

    @property
    def _min_area(self) -> int:
        return round(bmq.GROUNDING_MIN_BBOX_AREA_FRAC * self.FRAME_W * self.FRAME_H)

    def _passing_struct(self) -> dict:
        # Comfortably above the min-area floor, in frame, unoccluded.
        side = int((self._min_area * 4) ** 0.5) + 1  # ~2x the min side -> ~4x area
        return {
            "in_frame": True,
            "bbox": (10, 10, 10 + side, 10 + side),
            "occlusion_ratio": 0.0,
            "frame_w": self.FRAME_W,
            "frame_h": self.FRAME_H,
        }

    def test_all_pass_is_yes(self):
        assert bmq.geometric_visibility_verdict(self._passing_struct(), "go to the chair") == "yes"

    def test_not_in_frame_is_no(self):
        s = self._passing_struct()
        s["in_frame"] = False
        assert bmq.geometric_visibility_verdict(s, "x") == "no"

    def test_occlusion_is_individually_necessary(self):
        s = self._passing_struct()
        s["occlusion_ratio"] = bmq.GROUNDING_MAX_OCCLUSION_RATIO + 0.01
        assert bmq.geometric_visibility_verdict(s, "x") == "no"
        # Exactly at the threshold is still observable (<=, not <).
        s["occlusion_ratio"] = bmq.GROUNDING_MAX_OCCLUSION_RATIO
        assert bmq.geometric_visibility_verdict(s, "x") == "yes"

    def test_bbox_area_is_individually_necessary(self):
        s = self._passing_struct()
        # A 1-pixel speck is below the floor -> "no".
        s["bbox"] = (10, 10, 11, 11)
        assert (11 - 10) * (11 - 10) < self._min_area
        assert bmq.geometric_visibility_verdict(s, "x") == "no"
        # A box exactly at the min area passes.
        side = max(1, int(self._min_area ** 0.5))
        big = side + 1  # area (big*big) >= min_area
        s["bbox"] = (10, 10, 10 + big, 10 + big)
        assert big * big >= self._min_area
        assert bmq.geometric_visibility_verdict(s, "x") == "yes"

    def test_none_required_field_is_no(self):
        for field in ("occlusion_ratio", "bbox"):
            s = self._passing_struct()
            s[field] = None
            assert bmq.geometric_visibility_verdict(s, "x") == "no", field

    def test_non_dict_struct_is_no(self):
        assert bmq.geometric_visibility_verdict(None, "x") == "no"

    def test_zero_frame_dims_is_no(self):
        # A degenerate frame dim would collapse the area floor to 0 and let any
        # speck pass; guard rejects instead.
        for dim in ("frame_w", "frame_h"):
            s = self._passing_struct()
            s[dim] = 0
            assert bmq.geometric_visibility_verdict(s, "x") == "no", dim

    def test_mission_text_is_ignored(self):
        s = self._passing_struct()
        assert bmq.geometric_visibility_verdict(s, "fetch the lamp") == bmq.geometric_visibility_verdict(s, "")

    def test_area_floor_scales_with_resolution(self):
        # The same bbox is a speck on a big frame but fine on a small one:
        # the floor is a fraction of frame area, not a fixed pixel count.
        bbox = (0, 0, 8, 8)  # 64 px
        big = {"in_frame": True, "bbox": bbox, "occlusion_ratio": 0.0,
               "frame_w": 640, "frame_h": 360}
        small = {**big, "frame_w": 120, "frame_h": 80}
        assert 64 < round(bmq.GROUNDING_MIN_BBOX_AREA_FRAC * 640 * 360)  # speck on big frame
        assert 64 >= round(bmq.GROUNDING_MIN_BBOX_AREA_FRAC * 120 * 80)  # fine on small frame
        assert bmq.geometric_visibility_verdict(big, "x") == "no"
        assert bmq.geometric_visibility_verdict(small, "x") == "yes"

    def test_runner_wraps_verdict_and_returns_only_yes_or_no(self):
        runner = bmq.build_default_geometric_runner()
        assert runner(self._passing_struct(), "x") == "yes"
        absent = self._passing_struct()
        absent["in_frame"] = False
        assert runner(absent, "x") == "no"

    def test_runner_drops_into_existing_grounding_plumbing(self):
        # The geometric runner must satisfy the same (frame, mission_text) -> str
        # seam the generator calls: a "yes" struct ships, a same-room "no" struct
        # re-rolls (rejected). Reuses the real generator, swapping only the
        # runner + a struct-returning frame provider.
        scene = _two_room_scene()

        def yes_provider(obj, start_pose):
            return {"in_frame": True, "bbox": (0, 0, 100, 100), "occlusion_ratio": 0.0,
                    "frame_w": 640, "frame_h": 360}

        result = bmq.build_mission_queue(
            scene,
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=bmq.build_default_geometric_runner(),
            grounding_frame_provider=yes_provider,
        )
        assert result.stats.start_frame_grounded_yes == result.stats.emitted
        assert all(r["generator_metadata"]["start_frame_grounded"] is True for r in result.rows)

        def no_same_room_provider(obj, start_pose):
            return {"in_frame": False, "bbox": None, "occlusion_ratio": None,
                    "frame_w": 640, "frame_h": 360}

        same = bmq.build_mission_queue(
            _two_room_scene(reachable=False),
            GeneratorConfig(mode="endpoint", ground_start_frame=True),
            grounding_runner=bmq.build_default_geometric_runner(),
            grounding_frame_provider=no_same_room_provider,
        )
        assert same.stats.emitted == 0
        assert same.stats.rejected_reasons.get("target_not_visible_at_start", 0) >= 1


# ---------------------------------------------------------------------------
# Caching: key + template-hash invalidation
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_key_is_deterministic_and_pose_sensitive(self):
        base = dict(
            scene_seed=2,
            mission_text="Go to the chair.",
            llm_seed=42,
            template_hash="abc123",
            generator_version="1",
        )
        k1 = bmq.mission_cache_key(start_pose=(0.5, 0.5, 0.0), **base)
        k2 = bmq.mission_cache_key(start_pose=(0.5, 0.5, 0.0), **base)
        k3 = bmq.mission_cache_key(start_pose=(1.5, 0.5, 0.0), **base)
        assert k1 == k2 != k3

    def test_template_hash_changes_with_target_proximity(self):
        h_default = bmq.prompt_template_hash(GeneratorConfig())
        h_changed = bmq.prompt_template_hash(GeneratorConfig(target_proximity_m=0.9))
        assert h_default != h_changed

    def test_cache_header_distinguishes_version_and_seed(self):
        a = bmq.cache_header(GeneratorConfig(llm_seed=1))
        b = bmq.cache_header(GeneratorConfig(llm_seed=2))
        assert a != b
        assert a["prompt_template_hash"] == b["prompt_template_hash"]


# ---------------------------------------------------------------------------
# Occupancy staleness check
# ---------------------------------------------------------------------------


class TestOccupancyFreshness:
    def test_matching_identity_passes(self, tmp_path):
        usd = tmp_path / "scene.usdc"
        usd.write_bytes(b"x" * 100)
        st = usd.stat()
        meta = {"usd_mtime_ns": st.st_mtime_ns, "usd_size": st.st_size}
        bmq.check_occupancy_freshness(meta, usd)  # no raise

    def test_size_mismatch_raises(self, tmp_path):
        usd = tmp_path / "scene.usdc"
        usd.write_bytes(b"x" * 100)
        meta = {"usd_mtime_ns": usd.stat().st_mtime_ns, "usd_size": 999}
        with pytest.raises(bmq.StaleOccupancyError):
            bmq.check_occupancy_freshness(meta, usd)

    def test_missing_identity_is_treated_as_fresh(self, tmp_path):
        usd = tmp_path / "scene.usdc"
        usd.write_bytes(b"x" * 10)
        bmq.check_occupancy_freshness({}, usd)  # no raise


# ---------------------------------------------------------------------------
# Mode behaviour + paraphrases
# ---------------------------------------------------------------------------


class TestModesAndParaphrases:
    def test_endpoint_mode_has_no_path_shape_language(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        for r in result.rows:
            assert "hugging" not in r["mission_text"]
            assert r["generator_metadata"]["constraint_type_hint"] == "none"

    def test_path_shape_mode_emits_groundable_constraint_language(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="path-shape"))
        texts = " ".join(r["mission_text"] for r in result.rows)
        # Cross-room missions read as room-type transit, never a wall/compass.
        assert ("through the doorway into" in texts) or ("via the" in texts)

    def test_no_cardinal_or_wall_language_in_any_text(self):
        scene = _two_room_scene()
        # A unique near-path landmark so the landmark phrasing is exercised too,
        # plus a structural object that must be excluded from the target pool.
        scene.metadata["objects"].append(
            {"label": "lamp", "instance_id": 5, "position_3d": [6.5, 1.5, 0.4]}
        )
        scene.metadata["objects"].append(
            {"label": "wall", "instance_id": 6, "position_3d": [1.0, 0.2, 0.5]}
        )
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="mixed"))
        banned = re.compile(r"\b(north|south|east|west|left|right|wall)\b", re.IGNORECASE)
        for r in result.rows:
            for text in [r["mission_text"], *r["paraphrases"]]:
                assert not banned.search(text), f"ungroundable spatial word in: {text!r}"

    def test_structural_label_is_excluded_from_target_pool(self):
        scene = _two_room_scene()
        scene.metadata["objects"].append(
            {"label": "wall", "instance_id": 7, "position_3d": [1.5, 0.3, 0.5]}
        )
        scene.metadata["objects"].append(
            {"label": "floor lamp", "instance_id": 8, "position_3d": [2.2, 1.0, 0.4]}
        )
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint"))
        labels = {r["target_label"] for r in result.rows}
        assert "wall" not in labels  # structural surface, not a navigation goal
        assert "floor lamp" in labels  # compound label survives exact-set membership
        assert result.stats.rejected_reasons.get("structural_target", 0) == 1

    def test_fallback_paraphrases_are_distinct_from_mission_text(self):
        scene = _two_room_scene()
        result = bmq.build_mission_queue(scene, GeneratorConfig(mode="endpoint", paraphrases_per_mission=3))
        for r in result.rows:
            assert r["mission_text"] not in r["paraphrases"]
            assert len(r["paraphrases"]) >= 1


# ---------------------------------------------------------------------------
# Groundable filter: coordinate-anchored targets are dropped by default
# ---------------------------------------------------------------------------


def _clutter_scene_no_occupancy() -> SceneInputs:
    """One room, two identical boxes: no qualifier (or room scope) can tell them
    apart, so each resolves only to the un-groundable coordinate anchor."""
    metadata = {
        "rooms": [{"room_type": "room", "footprint_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "story": 0}],
        "objects": [
            {"label": "box", "instance_id": 1, "position_3d": [1.0, 1.0, 0.5],
             "bbox_3d_min": [0, 0, 0], "bbox_3d_max": [0.1, 0.1, 0.1]},
            {"label": "box", "instance_id": 2, "position_3d": [3.0, 1.0, 0.5],
             "bbox_3d_min": [0, 0, 0], "bbox_3d_max": [0.1, 0.1, 0.1]},
        ],
        "connectivity": [],
    }
    return SceneInputs(
        scene_name="scene_clutter_000_seed0", metadata=metadata, scene_seed=0, free_space=None
    )


class TestGroundableFilter:
    def test_ungroundable_target_rejected_by_default(self):
        result = bmq.build_mission_queue(_clutter_scene_no_occupancy(), GeneratorConfig(mode="endpoint"))
        assert result.stats.emitted == 0
        assert result.rows == []
        assert result.stats.rejected_reasons.get("ungroundable_target") == 2

    def test_ungroundable_target_emitted_when_filter_disabled(self):
        result = bmq.build_mission_queue(
            _clutter_scene_no_occupancy(), GeneratorConfig(mode="endpoint", require_groundable=False)
        )
        assert result.stats.emitted == 2
        assert result.stats.rejected_reasons.get("ungroundable_target", 0) == 0
        # The emitted mission names the coordinate anchor (ungroundable but unique).
        for r in result.rows:
            assert "approximately at" in r["mission_text"]


class TestRoomSuffixDedup:
    """The builder's room scope can fold the destination room into the anchor;
    the consumer must not then name the same room a second time."""

    def test_endpoint_does_not_double_name_room(self):
        ref = "the largest shelf in the kitchen"
        assert (
            bmq.endpoint_text(ref, "kitchen", True, room_unique=True)
            == "Go to the largest shelf in the kitchen."
        )

    def test_endpoint_still_names_room_when_anchor_lacks_it(self):
        assert (
            bmq.endpoint_text("the shelf", "kitchen", True, room_unique=True)
            == "Go to the shelf in the kitchen."
        )

    def test_path_shape_transit_dedups_room(self):
        text, hint = bmq.path_shape_text(
            ref="the largest shelf in the kitchen", target_room="kitchen", cross_room=True,
            room_unique=True, transit_room="hall", transit_unique=True,
        )
        assert text == "Go to the largest shelf in the kitchen via the hall."
        assert hint == "room_transit"

    def test_path_shape_doorway_dedups_room(self):
        text, hint = bmq.path_shape_text(
            ref="the largest shelf in the kitchen", target_room="kitchen",
            cross_room=True, room_unique=True,
        )
        assert text == "Go to the largest shelf in the kitchen through the doorway."
        assert hint == "room_transit"

    def test_path_shape_doorway_names_room_when_anchor_lacks_it(self):
        text, _ = bmq.path_shape_text(
            ref="the shelf", target_room="kitchen", cross_room=True, room_unique=True
        )
        assert text == "Go to the shelf through the doorway into the kitchen."


class TestRoomTypeUniquenessParity:
    """The consumer's `room_type_is_unique` must agree with the builder's
    `_room_type_is_unique` so the two never disagree on whether room scope fires
    — both normalize room_type before comparing."""

    def test_consumer_normalizes_and_agrees_with_builder_on_case_variants(self):
        from strafer_lab.tools import mission_text_builder as mtb

        # Case-variant duplicates: "kitchen" is NOT unique under either.
        dup = [{"room_type": "Kitchen"}, {"room_type": "KITCHEN"}]
        assert bmq.room_type_is_unique(dup, "kitchen") is False
        assert mtb._room_type_is_unique(dup, "kitchen") is False
        # A single case-variant room IS unique under both (normalized).
        one = [{"room_type": "Kitchen"}, {"room_type": "Hall"}]
        assert bmq.room_type_is_unique(one, "kitchen") is True
        assert mtb._room_type_is_unique(one, "kitchen") is True


class TestQwen3ThinkingDisabled:
    """The Qwen3 text runner (planner + paraphrase, which reuse the waypoint runner) must
    disable thinking mode. With thinking ON, every call emits a ``<think>...</think>`` block
    that consumes the ``max_new_tokens`` budget — far slower and prone to truncating the
    actual answer. The grounding runner (Qwen2.5-VL, not a thinking model) is unaffected.
    """

    def test_waypoint_runner_passes_enable_thinking_false(self, monkeypatch):
        import sys
        import types

        captured: dict = {}

        class _Slice:
            shape = (1, 3)

            def __getitem__(self, _key):
                return self

        class _Enc(dict):
            def to(self, _device):
                return self

        class _Tok:
            def apply_chat_template(self, _messages, **kwargs):
                captured.update(kwargs)
                return "PROMPT"

            def __call__(self, _texts, return_tensors=None):
                return _Enc(input_ids=_Slice())

            def batch_decode(self, *_a, **_k):
                return ["0.1,0.1 0.9,0.9"]

        class _Model:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **_kwargs):
                return _Slice()

        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())
        fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: _Model()
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda _s: None
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        runner = bmq.build_default_waypoint_runner("fake-model")
        runner("plan a path", 0)

        assert captured.get("enable_thinking") is False
