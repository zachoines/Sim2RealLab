"""Gym-side observation dump for bridge mode — the training half of parity (pure core).

The deployment lane already emits an assembled-observation JSONL from the
inference node (its ``obs_dump_path`` parameter). The bridge loop emits the
missing counterpart: per env step it evaluates the **same**
``mdp/observations.py`` term functions training assembles and appends one
schema-conformant line. Joining the two dumps on their shared sim clock
(``scripts/PARITY_SCHEMA.md`` in ``strafer_inference``) turns a train↔deploy
observation mismatch into a numeric, per-dimension failure instead of a silent
behavioural drift.

This module is the **Kit-free pure core**: JSONL serialization, schema ordering
against ``PolicyVariant``, the NaN policy, truncate-per-launch, and the
``assemble_observation`` scaling. It imports only numpy and the shared policy
interface, so the record/assembly logic is unit-testable without a live Kit
process. The **term evaluation** — which needs Isaac Lab and the live scene —
lives in the sibling :mod:`strafer_lab.bridge.obs_dump_terms`; that module calls
the real term functions and hands the raw values to :class:`ObsDumpWriter` here.
The split keeps the equivalence between the dump and the training terms
structural (verbatim function calls), pinned by a Kit test rather than a
re-derivation.

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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
)

# The observation fields evaluated from the live bridge scene by calling
# training's own term functions (in obs_dump_terms). Every other field a variant
# declares is NaN-filled here. Keyed by ``ObsField.key`` so the rule holds across
# variants: the goal-shaped triplet is ``goal_*`` on NOCAM / DEPTH and
# ``subgoal_*`` on the subgoal variants, and neither is in this set.
COMPUTABLE_KEYS: frozenset[str] = frozenset(
    {
        "imu_accel",
        "imu_gyro",
        "encoder_vels_ticks",
        "body_velocity_xy",
        "depth_image",
    }
)

# Back-compat alias for the private spelling the pure tests reference.
_COMPUTABLE_KEYS = COMPUTABLE_KEYS


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
        if field.key in COMPUTABLE_KEYS:
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


class ObsDumpWriter:
    """Append one schema-conformant obs line per call, from raw term values.

    Kit-free: it takes already-evaluated term values (produced under Kit by
    :mod:`strafer_lab.bridge.obs_dump_terms`) and owns only assembly +
    serialization. The file is opened line-buffered and truncated per launch —
    mirroring the node's discipline: under sim time a relaunch resets the clock,
    and concatenating two runs would silently contaminate the sim-time join.
    """

    def __init__(self, path: str | Path, variant: PolicyVariant) -> None:
        self.variant = variant
        # buffering=1 → flush on each trailing newline; "w" → truncate per run.
        self._fh = open(Path(path), "w", buffering=1)

    def write(self, t_sim: float, term_values: dict[str, np.ndarray]) -> None:
        """Assemble ``term_values`` into the variant vector and append one line."""
        vec = build_obs_vector(self.variant, term_values)
        self._fh.write(json.dumps(obs_record(self.variant, t_sim, vec)) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ObsDumpWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
