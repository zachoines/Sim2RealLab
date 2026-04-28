"""Frame-extras helper for the sim-in-the-loop reachability dataset.

``PerceptionFrameWriter.save_frame(extras=...)`` accepts an arbitrary
mapping that gets merged into each frame's ``frames.jsonl`` record.
This module produces that mapping for sim-in-the-loop runs so every
frame carries the mission context plus the eventual reachability label,
without duplicating per-frame fields the writer already records
(robot pose, scene_name, image dims).

Schema additions per frame::

    {
      "mission_id": "kitchen_01__chair__42",
      "target_label": "Chair",
      "target_instance_id": 42,
      "target_position_3d": [x, y, z],
      "target_prim_path": "/World/Chair_42",
      "reachability": true,                # set when the episode ends
      "mission_state": "succeeded",        # final_state from MissionStatusSnapshot
      "mission_error_code": "",
      "mission_elapsed_s": 12.4
    }

Pure Python — no Isaac Sim, no ROS. Importable from .venv_vlm.
"""

from __future__ import annotations

from typing import Any, Mapping

from strafer_lab.sim_in_the_loop.mission import MissionSpec


def make_episode_extras(
    *,
    spec: MissionSpec,
    reachability: bool | None,
    mission_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-frame ``extras`` dict for a sim-in-the-loop episode.

    Parameters
    ----------
    spec:
        The :class:`MissionSpec` that produced the current episode. Its
        identifier and target metadata get copied verbatim into every
        frame so the dataset is self-describing without joining back to
        ``scene_metadata.json``.
    reachability:
        ``True`` if the executor reported ``final_state == "succeeded"``
        for this mission, ``False`` if it reported failure or timed out,
        ``None`` while the mission is still in flight (the harness
        passes ``None`` on every frame except the last; on the last
        frame it knows the outcome and passes the bool).
    mission_status:
        Optional mapping with the executor's terminal status fields
        (``state``, ``error_code``, ``elapsed_s``). Typically obtained
        from ``MissionStatus.as_dict()`` in
        :mod:`strafer_lab.sim_in_the_loop.harness`. Missing keys are
        silently dropped.
    """

    extras: dict[str, Any] = {
        "mission_id": spec.mission_id,
        "target_label": spec.target_label,
        "target_instance_id": spec.target_instance_id,
        "target_position_3d": list(spec.target_position_3d),
    }
    if spec.target_prim_path:
        extras["target_prim_path"] = spec.target_prim_path
    if spec.target_semantic_tags:
        extras["target_semantic_tags"] = list(spec.target_semantic_tags)
    if spec.target_room_idx is not None:
        extras["target_room_idx"] = int(spec.target_room_idx)

    if reachability is not None:
        extras["reachability"] = bool(reachability)

    if mission_status is not None:
        for key in ("state", "error_code", "elapsed_s", "message"):
            if key in mission_status:
                extras[f"mission_{key}"] = mission_status[key]

    return extras
