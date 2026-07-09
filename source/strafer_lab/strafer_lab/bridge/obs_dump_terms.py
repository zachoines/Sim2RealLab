"""Kit-side term evaluation for the bridge observation dump.

Pairs with the Kit-free :mod:`strafer_lab.bridge.obs_dump` core (serialization,
schema ordering, NaN policy, ``assemble_observation`` scaling). This module owns
the half that needs a live scene: it calls the **actual**
``mdp/observations.py`` term functions training assembles — verbatim, no
re-derived math — against the bridge env's handles, and hands the raw values to
:class:`~strafer_lab.bridge.obs_dump.ObsDumpWriter`.

Evaluating the training terms directly (rather than the bridge env's own
observation manager) is deliberate: the bridge cfg runs a *realistic* profile —
corruption noise on, and a final-goal referent rather than a rolling subgoal —
so its obs tensor is not the quantity the deployed policy consumes.

Noise discipline — the dump compares **clean** signals on both sides. The term
functions return clean values; Isaac Lab injects observation noise only in the
``ObservationManager`` (``ObsTermCfg.noise``, and only when the group's
``enable_corruption`` is set). Calling the terms directly bypasses the manager,
so no domain-randomization noise is applied — matching the deployment contract,
where the node adds none (``strafer_inference`` ``obs_pipeline``: "inference
adds none"). No term function injects noise internally (each returns raw
sensor/FK output and defers ``.noise`` / ``.scale`` to the manager), so there is
nothing to disable here.

The Isaac Lab / mdp imports are deferred to :func:`evaluate_obs_terms`, so this
module itself imports Kit-free — only the actual evaluation needs a live env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from strafer_shared.constants import DEPTH_MAX
from strafer_shared.policy_interface import PolicyVariant

from strafer_lab.bridge.obs_dump import ObsDumpWriter


def _term_to_numpy(tensor) -> np.ndarray:
    """Take env row 0 of a ``(num_envs, D)`` term output as a 1-D numpy array.

    Sim-in-the-loop forces ``num_envs=1``; the term functions still return a
    batch axis, so index it off before serialization.
    """
    return np.asarray(tensor[0].detach().cpu().numpy(), dtype=np.float32)


def evaluate_obs_terms(env, variant: PolicyVariant) -> dict[str, np.ndarray]:
    """Evaluate training's computable obs terms against the live bridge scene.

    Calls the real ``mdp/observations.py`` functions (no re-derivation): the two
    IMU terms against the ``d555_imu`` sensor, the wheel-encoder velocities and
    the encoder-FK body velocity against the ``robot`` articulation, and — for a
    depth variant — the depth term against the ``d555_camera`` policy camera. The
    term functions read ``env.scene`` handles by name; an entity cfg needs only
    its name to index the scene, so no manager resolution is required.

    Returns raw (unscaled) values keyed by ``ObsField.key`` — exactly the
    computable keys the variant declares. The pure core NaN-fills the rest.
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
    if any(f.key == "depth_image" for f in variant.fields):
        # max_depth from the shared constant, matching the training term's wired
        # param; nearfield clip/fill keep the term defaults (which the training
        # wiring also leaves untouched).
        values["depth_image"] = _term_to_numpy(
            depth_image(env, SceneEntityCfg("d555_camera"), max_depth=DEPTH_MAX)
        )
    return values


class BridgeObsDumper:
    """Evaluate the terms against the live env and append one obs line per step.

    Thin façade over the Kit-side :func:`evaluate_obs_terms` and the Kit-free
    :class:`~strafer_lab.bridge.obs_dump.ObsDumpWriter`.
    """

    def __init__(self, path: str | Path, variant: PolicyVariant) -> None:
        self.variant = variant
        self._writer = ObsDumpWriter(path, variant)

    def write(self, env, t_sim: float) -> None:
        """Evaluate the terms against ``env`` and append one line at ``t_sim``."""
        self._writer.write(t_sim, evaluate_obs_terms(env, self.variant))

    def close(self) -> None:
        self._writer.close()

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
