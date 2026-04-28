"""Tests for strafer_lab.sim_in_the_loop.extras.make_episode_extras.

Pure Python — runs in .venv_vlm via the strafer_lab namespace stub.
"""

from __future__ import annotations

from strafer_lab.sim_in_the_loop.extras import make_episode_extras
from strafer_lab.sim_in_the_loop.mission import MissionSpec


def _spec(**overrides) -> MissionSpec:
    base = dict(
        mission_id="kitchen_01__chair__1",
        scene_name="kitchen_01",
        target_label="Chair",
        target_instance_id=1,
        target_position_3d=(1.0, 2.0, 0.0),
        target_room_idx=0,
        raw_command="go to the chair",
        target_semantic_tags=("seating", "wood"),
        target_prim_path="/World/Chair_1",
    )
    base.update(overrides)
    return MissionSpec(**base)


class TestExtrasMissionMetadata:
    def test_includes_mission_id(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["mission_id"] == "kitchen_01__chair__1"

    def test_includes_target_label_and_instance(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["target_label"] == "Chair"
        assert extras["target_instance_id"] == 1

    def test_target_position_is_a_list_for_jsonl(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["target_position_3d"] == [1.0, 2.0, 0.0]
        assert isinstance(extras["target_position_3d"], list)

    def test_optional_prim_path_included(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["target_prim_path"] == "/World/Chair_1"

    def test_optional_prim_path_omitted_when_missing(self):
        extras = make_episode_extras(
            spec=_spec(target_prim_path=None), reachability=None,
        )
        assert "target_prim_path" not in extras

    def test_optional_semantic_tags_included(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["target_semantic_tags"] == ["seating", "wood"]

    def test_optional_semantic_tags_omitted_when_empty(self):
        extras = make_episode_extras(
            spec=_spec(target_semantic_tags=()), reachability=None,
        )
        assert "target_semantic_tags" not in extras

    def test_optional_room_idx_included(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert extras["target_room_idx"] == 0

    def test_optional_room_idx_omitted_when_none(self):
        extras = make_episode_extras(
            spec=_spec(target_room_idx=None), reachability=None,
        )
        assert "target_room_idx" not in extras


class TestExtrasReachability:
    def test_none_reachability_omits_field(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        assert "reachability" not in extras

    def test_true_reachability(self):
        extras = make_episode_extras(spec=_spec(), reachability=True)
        assert extras["reachability"] is True

    def test_false_reachability(self):
        extras = make_episode_extras(spec=_spec(), reachability=False)
        assert extras["reachability"] is False


class TestExtrasMissionStatus:
    def test_status_dict_promoted_with_mission_prefix(self):
        extras = make_episode_extras(
            spec=_spec(),
            reachability=True,
            mission_status={
                "state": "succeeded",
                "error_code": "",
                "elapsed_s": 12.4,
                "message": "ok",
            },
        )
        assert extras["mission_state"] == "succeeded"
        assert extras["mission_error_code"] == ""
        assert extras["mission_elapsed_s"] == 12.4
        assert extras["mission_message"] == "ok"

    def test_partial_status_dict_only_promotes_present_keys(self):
        extras = make_episode_extras(
            spec=_spec(),
            reachability=False,
            mission_status={"state": "failed", "error_code": "nav_timeout"},
        )
        assert extras["mission_state"] == "failed"
        assert extras["mission_error_code"] == "nav_timeout"
        assert "mission_elapsed_s" not in extras
        assert "mission_message" not in extras

    def test_no_status_omits_mission_fields(self):
        extras = make_episode_extras(spec=_spec(), reachability=None)
        for key in extras:
            assert not key.startswith("mission_state")
            assert not key.startswith("mission_error_code")
