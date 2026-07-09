"""Gym-side observation dump for bridge mode — the training half of parity.

The deployment lane already emits an assembled-observation JSONL from the
inference node (its ``obs_dump_path`` parameter). This module is the missing
counterpart: driven from the sim-in-the-loop bridge loop, it evaluates the
**same** ``mdp/observations.py`` term functions the training env assembles
against the live bridge scene handles and appends one schema-conformant line
per env step. Joining the two dumps on their shared sim clock
(``scripts/PARITY_SCHEMA.md`` in ``strafer_inference``) turns a train↔deploy
observation mismatch into a numeric, per-dimension failure instead of a
silent behavioural drift.

Why the term functions and not the bridge env's own observation manager: the
bridge cfg runs a *realistic* observation profile — corruption noise on, and a
final-goal referent rather than a rolling subgoal — so its obs tensor is not
the quantity the deployed policy consumes. Calling the term functions directly
reproduces the clean, deploy-equivalent value; ``assemble_observation`` then
applies exactly the per-field scale and ordering the node uses, so the whole
assembly path is exercised end to end.

Fields that cannot be computed from the gym env alone are NaN-filled:

  - the referent-derived triplet (``goal_relative`` / ``goal_distance`` /
    ``goal_heading_to_goal``, or their ``subgoal_*`` equivalents) — in bridge
    mode the rolling subgoal is picked by the Jetson generator off its own
    planned path, not by this env;
  - ``last_action`` — the node's own previous policy output, internal to the
    node and not observable from the bridge.

The parity comparator masks NaN dimensions from the bound check and reports
them as masked (see ``strafer_inference.parity.compute_obs_parity``), so those
dims neither pass nor fail — they are accounted for, not hidden.

The Kit-bound term evaluation is deferred to :meth:`BridgeObsDumper.write`;
importing this module pulls only numpy and the shared policy interface, so the
record/assembly logic is unit-testable without a live Kit process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from strafer_shared.constants import DEPTH_MAX
from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
)

# The observation fields this dumper evaluates from the live bridge scene by
# calling training's own term functions. Every other field a variant declares
# is NaN-filled (see the module docstring). Keyed by ``ObsField.key`` so the
# rule holds across variants: the goal-shaped triplet is ``goal_*`` on NOCAM /
# DEPTH and ``subgoal_*`` on the subgoal variants, and neither is in this set.
_COMPUTABLE_KEYS: frozenset[str] = frozenset(
    {
        "imu_accel",
        "imu_gyro",
        "encoder_vels_ticks",
        "body_velocity_xy",
        "depth_image",
    }
)


def build_raw_term_dict(
    variant: PolicyVariant,
    term_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Assemble the raw dict for :func:`assemble_observation`.

    ``term_values`` carries the raw (unscaled) output of each computable term
    keyed by its ``ObsField.key``. Every computable field the variant declares
    must be present with the right dimension; every non-computable field is
    filled with NaN so the assembled vector keeps the variant's full length and
    ordering while marking the dims the gym side cannot compute.
    """
    raw: dict[str, np.ndarray] = {}
    for field in variant.fields:
        if field.key in _COMPUTABLE_KEYS:
            arr = np.asarray(term_values[field.key], dtype=np.float32).ravel()
            if arr.shape[0] != field.dims:
                raise ValueError(
                    f"term '{field.key}': got {arr.shape[0]} dims, "
                    f"expected {field.dims} for {variant.name}"
                )
            raw[field.key] = arr
        else:
            raw[field.key] = np.full(field.dims, np.nan, dtype=np.float32)
    return raw


def build_obs_vector(
    variant: PolicyVariant,
    term_values: dict[str, np.ndarray],
) -> np.ndarray:
    """Full variant-ordered, scaled obs vector: computed terms + NaN fills.

    Delegates scaling, ordering, and concatenation to
    :func:`assemble_observation` — the exact assembly the inference node runs —
    so the computed dims match the deployed vector to float32 and the NaN dims
    ride along in their contracted positions.
    """
    return assemble_observation(build_raw_term_dict(variant, term_values), variant)


def obs_record(
    variant: PolicyVariant,
    t_sim: float,
    obs_vector: np.ndarray,
) -> dict:
    """Build one ``PARITY_SCHEMA.md`` obs-dump record.

    ``referent`` is always null: the bridge env does not own the rolling
    subgoal the goal-shaped fields refer to (it lives in the Jetson generator),
    so there is no map-frame pose to record here — the node dump carries it.
    """
    return {
        "t_sim": float(t_sim),
        "variant": variant.name,
        "obs": np.asarray(obs_vector, dtype=np.float32).tolist(),
        "referent": None,
    }


def _term_to_numpy(tensor) -> np.ndarray:
    """Take env row 0 of a ``(num_envs, D)`` term output as a 1-D numpy array.

    Sim-in-the-loop forces ``num_envs=1``; the term functions still return a
    batch axis, so index it off before serialization.
    """
    return np.asarray(tensor[0].detach().cpu().numpy(), dtype=np.float32)


class BridgeObsDumper:
    """Append one schema-conformant obs line per bridge step.

    The file is opened line-buffered and truncated per launch — mirroring the
    node's discipline: under sim time a relaunch resets the clock, and
    concatenating two runs would silently contaminate the sim-time join.
    """

    def __init__(self, path: str | Path, variant: PolicyVariant) -> None:
        self.variant = variant
        self._has_depth = any(f.key == "depth_image" for f in variant.fields)
        # buffering=1 → flush on each trailing newline; "w" → truncate per run.
        self._fh = open(Path(path), "w", buffering=1)

    def _evaluate_terms(self, env) -> dict[str, np.ndarray]:
        """Evaluate the computable training terms against the live scene.

        Kit-bound imports are deferred to call time so importing this module
        stays Kit-free. The term functions read ``env.scene`` handles by name;
        an entity cfg needs only its name to index the scene, so no manager
        resolution is required here.
        """
        from isaaclab.managers import SceneEntityCfg

        from strafer_lab.tasks.navigation.mdp import (
            body_velocity_xy,
            depth_image,
            imu_angular_velocity,
            imu_linear_acceleration,
            wheel_encoder_velocities,
        )

        imu_cfg = SceneEntityCfg("d555_imu")
        values: dict[str, np.ndarray] = {
            "imu_accel": _term_to_numpy(imu_linear_acceleration(env, imu_cfg)),
            "imu_gyro": _term_to_numpy(imu_angular_velocity(env, imu_cfg)),
            "encoder_vels_ticks": _term_to_numpy(wheel_encoder_velocities(env)),
            "body_velocity_xy": _term_to_numpy(body_velocity_xy(env)),
        }
        if self._has_depth:
            # max_depth from the shared constant, matching the training term's
            # wired param; nearfield clip/fill keep the term defaults (which the
            # training wiring also leaves untouched).
            values["depth_image"] = _term_to_numpy(
                depth_image(env, SceneEntityCfg("d555_camera"), max_depth=DEPTH_MAX)
            )
        return values

    def write(self, env, t_sim: float) -> None:
        """Evaluate the terms and append one obs line stamped at ``t_sim``."""
        vec = build_obs_vector(self.variant, self._evaluate_terms(env))
        self._fh.write(json.dumps(obs_record(self.variant, t_sim, vec)) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "BridgeObsDumper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_bridge_obs_dumper(
    path: str, variant_name: str
) -> Optional[BridgeObsDumper]:
    """Build a dumper for ``path``, or ``None`` when disabled (empty path).

    ``variant_name`` must be a :class:`PolicyVariant` name and must match the
    variant the inference node emits, or the parity join rejects the pair on a
    variant mismatch.
    """
    if not path or not path.strip():
        return None
    return BridgeObsDumper(path.strip(), PolicyVariant[variant_name])
