"""Parity tests for the block reduction that puts the deploy field in training.

The policy camera renders at the deploy stream's resolution and
``depth_image`` reduces it to the policy grid, so the clean training field is
the deploy field by construction rather than by resemblance. That only holds
while the two reductions are the same operator, and they are written in
different array libraries against different call signatures, so the agreement
is pinned byte-for-byte rather than to a tolerance.

Byte-equality is the right bar because the reduction is exact arithmetic on
the same 64 inputs: any difference is a different operator, not a rounding
difference. The even-count median is where the two libraries part company --
numpy averages the two middle values, ``torch.median`` returns the
lower-middle one -- and on a block that straddles a depth discontinuity those
differ by the size of the discontinuity.

The deployment half is ``strafer_inference.obs_pipeline``, reached by path so
the gate runs on the training host rather than skipping there.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from strafer_lab.tasks.navigation.d555_cfg import make_d555_camera_cfg
from strafer_lab.tasks.navigation.mdp.observations import depth_image
from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_WIDTH,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
)


def _load_obs_pipeline():
    """Load the deploy pipeline from its file.

    By path rather than by import: the package is a ROS package that is not on
    this environment's path, and putting it there would claim names like
    ``test`` and ``config`` for the rest of the session. The module is
    deliberately rclpy-free, so the file loads on its own.
    """
    path = (
        Path(__file__).resolve().parents[3]
        / "strafer_ros" / "strafer_inference" / "strafer_inference"
        / "obs_pipeline.py"
    )
    if not path.exists():
        pytest.skip(f"deploy pipeline not present at {path}")
    spec = importlib.util.spec_from_file_location("_obs_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs_pipeline = _load_obs_pipeline()

_TRIALS = 20
_BLOCK = PERCEPTION_HEIGHT // DEPTH_HEIGHT


def _term(raw: np.ndarray) -> np.ndarray:
    """Run the real observation term over one rendered field.

    The term reads its image off a sensor in the scene, so the sensor is the
    only thing stubbed; everything downstream of the read is the shipped code
    path, which is what the byte-equality claim has to be about.
    """
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            output={"distance_to_image_plane": torch.from_numpy(raw)[None, ..., None]}
        )
    )
    env = types.SimpleNamespace(
        scene=types.SimpleNamespace(sensors={"d555_camera": sensor})
    )
    cfg = types.SimpleNamespace(name="d555_camera")
    return depth_image(env, cfg).numpy().reshape(-1)


def _rendered_field(rng: np.random.Generator) -> np.ndarray:
    """A deploy-resolution field carrying every value the renderer emits.

    +inf is the frustum cull, NaN is the real sensor's dropout, and -inf is
    neither but is what a sign error on the cull would produce, so it is
    included to pin that all three take the same branch.
    """
    field = rng.uniform(0.0, DEPTH_MAX * 1.5, (PERCEPTION_HEIGHT, PERCEPTION_WIDTH))
    field = field.astype(np.float32)
    field[rng.random(field.shape) < 0.05] = np.inf
    field[rng.random(field.shape) < 0.05] = np.nan
    field[rng.random(field.shape) < 0.02] = -np.inf

    # A block straddling the near clip: half its pixels below the sensor's
    # minimum range, half above. Its median decides the block's near-field
    # class, so the two libraries' even-count medians must land on the same
    # side of the threshold.
    straddle = np.concatenate(
        [np.full(_BLOCK * _BLOCK // 2, DEPTH_MIN * 0.25, dtype=np.float32),
         np.full(_BLOCK * _BLOCK // 2, DEPTH_MIN * 4.0, dtype=np.float32)]
    ).reshape(_BLOCK, _BLOCK)
    field[:_BLOCK, :_BLOCK] = straddle

    # A second block whose two middle values bracket the threshold, so the
    # averaged median and the lower-middle median fall on opposite sides.
    bracket = np.concatenate(
        [np.full(_BLOCK * _BLOCK // 2, DEPTH_MIN * 0.9, dtype=np.float32),
         np.full(_BLOCK * _BLOCK // 2, DEPTH_MIN * 1.2, dtype=np.float32)]
    ).reshape(_BLOCK, _BLOCK)
    field[_BLOCK:2 * _BLOCK, _BLOCK:2 * _BLOCK] = bracket
    return field


def test_the_reduction_is_byte_identical_to_the_deploy_pipeline():
    """Training's reduced field equals deployment's, bit for bit.

    Run over independent random fields rather than one, because the operators
    only diverge on blocks whose middle two values differ and a single field
    could miss them; the constructed straddle and bracket blocks guarantee at
    least two such blocks per trial.
    """
    rng = np.random.default_rng(20260919)
    for trial in range(_TRIALS):
        raw = _rendered_field(rng)
        expected = obs_pipeline.downsample_depth(raw)
        got = _term(raw)
        assert got.shape == expected.shape, (
            f"trial {trial}: training reduced to {got.shape}, deployment to "
            f"{expected.shape}"
        )
        mismatched = int((got != expected).sum())
        assert mismatched == 0, (
            f"trial {trial}: {mismatched}/{expected.size} policy pixels differ "
            f"between the training reduction and the deploy reduction; the "
            f"largest gap is "
            f"{np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64)))}"
        )


def test_a_field_already_on_the_policy_grid_is_not_reduced_again():
    """The reduction is gated on the rendered shape, not on the term.

    A camera cfg built at the policy dimensions still produces a usable
    observation, which is what keeps the pre-reduction render reachable from a
    scratch config without a cfg field selecting it.
    """
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.0, DEPTH_MAX, (DEPTH_HEIGHT, DEPTH_WIDTH)).astype(np.float32)
    got = _term(raw)
    assert got.shape == (DEPTH_HEIGHT * DEPTH_WIDTH,)

    expected = np.clip(raw, 0.0, DEPTH_MAX).reshape(-1)
    expected = np.where(expected < DEPTH_MIN, np.float32(0.2), expected)
    assert np.array_equal(got, np.clip(expected, 0.0, DEPTH_MAX))


def test_the_even_count_median_is_what_makes_the_two_agree():
    """Mutation guard: the lower-middle median breaks the parity.

    Stated as its own assertion so the byte-equality test above is known to be
    sensitive to the one choice it exists to pin, rather than passing because
    both sides happen to be smooth.
    """
    rng = np.random.default_rng(20260919)
    raw = _rendered_field(rng)
    expected = obs_pipeline.downsample_depth(raw)

    depth = torch.from_numpy(raw)[None, ..., None]
    depth = torch.where(
        torch.isinf(depth) | torch.isnan(depth),
        torch.full_like(depth, DEPTH_MAX),
        depth,
    )
    blocks = depth.reshape(
        1, DEPTH_HEIGHT, _BLOCK, DEPTH_WIDTH, _BLOCK, -1
    ).permute(0, 1, 3, 5, 2, 4).reshape(
        1, DEPTH_HEIGHT, DEPTH_WIDTH, -1, _BLOCK * _BLOCK
    )
    mutated = blocks.median(dim=-1).values
    mutated = torch.where(
        mutated < DEPTH_MIN, torch.full_like(mutated, 0.2), mutated
    )
    mutated = torch.clamp(mutated, 0.0, DEPTH_MAX).numpy().reshape(-1)

    assert not np.array_equal(mutated, expected), (
        "torch.median reproduced the deploy reduction, so the byte-equality "
        "test cannot be detecting the even-count median at all"
    )


def test_the_policy_camera_renders_the_field_the_reduction_expects():
    """The camera and the reduction are one decision, so they are pinned together.

    The composition contract cannot reach this: its serializer hashes the
    managers and takes only ``num_envs`` and ``env_spacing`` off the scene, so
    a camera resolution change moves no golden. Without this assertion the
    render and the reduction could drift apart silently, and the term would
    quietly stop reducing.
    """
    cam = make_d555_camera_cfg(data_types=("distance_to_image_plane",))
    assert (cam.height, cam.width) == (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), (
        f"the policy camera renders {cam.height}x{cam.width}; the reduction "
        f"is gated on {PERCEPTION_HEIGHT}x{PERCEPTION_WIDTH} and would pass "
        f"the raw field through"
    )
    assert cam.height == _BLOCK * DEPTH_HEIGHT and cam.width == _BLOCK * DEPTH_WIDTH, (
        "the block ratio is not an exact integer in both axes"
    )
