"""Parity tests for the near-field depth convention.

One pixel the camera cannot resolve must mean the same thing on both sides of
the sim-to-real boundary. Training assembles the policy's depth in two stages —
``depth_image`` fills the near field, then ``DepthNoiseModel`` adds realism —
while deployment assembles it in one, ``downsample_depth``. The two stages ran
opposite conventions: the term wrote the near fill and the noise model inverted
it to the far clamp, so roughly a fifth of every training frame read open floor
where the robot reads an obstacle.

These tests pin the reconciliation:

1. The threshold and the value written are the same number, which makes the
   ``too_close`` comparison idempotent rather than a coin flip.
2. An image the observation term has already filled survives the noise model
   with its near-field class intact.
3. The training and deployment paths agree on the near-field class of the same
   scene, so the two conventions cannot drift apart silently again.

The deployment half is ``strafer_inference.obs_pipeline``, which is kept
rclpy-free for exactly this reason; it is reached by path so the parity gate
runs on the training host rather than skipping there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from strafer_lab.tasks.navigation.mdp.noise_models import (
    DepthNoiseModel,
    DepthNoiseModelCfg,
)
from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_NEARFIELD_FILL,
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


def _noise_model(
    num_envs: int = 1, hole_probability: float = 0.0, **convention
) -> DepthNoiseModel:
    cfg = DepthNoiseModelCfg(
        hole_probability=hole_probability,
        min_range=DEPTH_NEARFIELD_FILL,
        max_range=DEPTH_MAX,
        height=DEPTH_HEIGHT,
        width=DEPTH_WIDTH,
        latency_steps=0,
        **convention,
    )
    return DepthNoiseModel(cfg, num_envs, "cpu")


def _training_nearfield_fill(depth: torch.Tensor) -> torch.Tensor:
    """The observation term's near-field stage, on an already-flat image.

    Mirrors ``observations.depth_image`` without a camera sensor to read.
    """
    depth = torch.where(
        depth < DEPTH_MIN, torch.full_like(depth, DEPTH_NEARFIELD_FILL), depth
    )
    return torch.clamp(depth, 0.0, DEPTH_MAX)


def test_the_threshold_sits_on_the_value_the_pipeline_writes():
    """The near fill and the noise model's floor are one number.

    This is the coincidence that made the comparison a coin flip, and writing
    the fill value back is what makes it harmless. If the two ever diverge,
    the convention needs restating rather than retuning.
    """
    assert DEPTH_NEARFIELD_FILL == DepthNoiseModelCfg().min_range, (
        "the value the observation term writes and the depth the noise model "
        "treats as unresolvable must be the same number"
    )


def test_the_shipped_noise_models_declare_the_policy_frame():
    """Every configured tier's declared shape is the camera's, not merely its area.

    The model can only check that the declared pixel count matches the frame it
    receives, so a transposed shape would satisfy it and median a different
    neighbourhood. The orientation is pinned here instead, where the shipped
    contracts are in hand.
    """
    from strafer_lab.tasks.navigation.sim_real_cfg import (
        REAL_ROBOT_CONTRACT,
        ROBUST_TRAINING_CONTRACT,
        get_depth_noise,
    )

    for name, contract in (("real", REAL_ROBOT_CONTRACT),
                           ("robust", ROBUST_TRAINING_CONTRACT)):
        cfg = get_depth_noise(contract)
        assert (cfg.height, cfg.width) == (DEPTH_HEIGHT, DEPTH_WIDTH), (
            f"the {name} tier declares {cfg.height}x{cfg.width}, not the "
            f"policy frame {DEPTH_HEIGHT}x{DEPTH_WIDTH}"
        )


def test_an_already_filled_image_survives_the_noise_model():
    """The near-field class the term created reaches the policy intact."""
    model = _noise_model()
    filled = _training_nearfield_fill(
        torch.full((1, DEPTH_HEIGHT * DEPTH_WIDTH), 0.1)
    )
    assert torch.allclose(filled, torch.full_like(filled, DEPTH_NEARFIELD_FILL))

    for _ in range(50):
        noisy = model(filled.clone())
        assert bool((noisy >= DEPTH_NEARFIELD_FILL).all())
        assert not bool((noisy >= DEPTH_MAX - 1e-3).any()), (
            "a pixel the term marked as near reached the far clamp"
        )


def test_both_paths_agree_on_the_near_field_class():
    """One scene, both pipelines, the same near-field pixels.

    The scene is constant within each 8x8 deployment block, so the block median
    returns the block's own value and the two paths are comparable pixel for
    pixel: training renders one ray per policy pixel, deployment reduces 64.
    """
    rng = np.random.default_rng(0)
    blocks = rng.uniform(0.05, 5.0, size=(DEPTH_HEIGHT, DEPTH_WIDTH))
    blocks[20:30, 10:40] = 0.15  # a contiguous near-field band
    raw = np.kron(blocks, np.ones((8, 8), dtype=np.float64)).astype(np.float32)
    assert raw.shape == (PERCEPTION_HEIGHT, PERCEPTION_WIDTH)

    deployed = obs_pipeline.downsample_depth(raw)
    trained = _training_nearfield_fill(
        torch.from_numpy(blocks.astype(np.float32)).reshape(1, -1)
    )

    near_deployed = deployed < DEPTH_MIN + 1e-6
    near_trained = (trained[0].numpy() < DEPTH_MIN + 1e-6)
    assert np.array_equal(near_deployed, near_trained), (
        "the two paths disagree on which pixels are near field"
    )
    assert np.allclose(deployed, trained[0].numpy(), atol=1e-6), (
        "the two paths disagree on the value written"
    )

    # The noise model must not reclassify what both paths agreed on.
    noisy = _noise_model()(trained.clone())[0].numpy()
    assert np.array_equal(noisy < DEPTH_MIN + 1e-6, near_trained), (
        "the realism noise moved pixels in or out of the near-field class"
    )


def test_an_isolated_unresolvable_pixel_reaches_the_policy_as_its_surface():
    """Both paths outvote one failed pixel rather than reading it as space.

    This is the property the two share. They are not the same operation — the
    deploy reduction medians a pixel's own 8x8 raw footprint with invalids
    counted as the far clamp, while training medians the valid neighbours among
    adjacent policy pixels — so the claim under test is the effect, not the
    arithmetic. Where a whole neighbourhood fails they diverge by design, and
    ``test_a_saturated_neighbourhood_diverges_by_design`` records that.
    """
    surface = 3.0
    raw = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), surface, dtype=np.float32)
    raw[0, 0] = np.nan  # one unresolvable source pixel inside a valid block
    deployed = obs_pipeline.downsample_depth(raw)
    assert deployed[0] == pytest.approx(surface), (
        "deployment let one invalid source pixel through its block median"
    )

    model = _noise_model(hole_probability=0.0)
    frame = torch.full((1, DEPTH_HEIGHT * DEPTH_WIDTH), surface)
    invalid = torch.zeros_like(frame, dtype=torch.bool)
    invalid[0, 0] = True
    trained = model._neighbourhood_median(frame.clone(), invalid)
    assert trained[0, 0].item() == pytest.approx(surface), (
        "training let one invalid pixel read as something other than its surface"
    )


def test_a_saturated_neighbourhood_diverges_by_design():
    """Where nothing valid is left to read, the two paths part company.

    Deployment reads the far clamp, training the near fill. Recorded rather
    than reconciled: the near fill is the conservative reading for a policy
    that has to avoid what it cannot resolve, and the case needs every pixel of
    a neighbourhood to fail at once.
    """
    raw = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), np.nan, dtype=np.float32)
    assert obs_pipeline.downsample_depth(raw)[0] == pytest.approx(DEPTH_MAX)

    model = _noise_model(hole_probability=0.0)
    frame = torch.full((1, DEPTH_HEIGHT * DEPTH_WIDTH), 3.0)
    invalid = torch.ones_like(frame, dtype=torch.bool)
    filled = model._neighbourhood_median(frame.clone(), invalid)
    assert filled[0, 0].item() == pytest.approx(DEPTH_NEARFIELD_FILL)


def test_the_retired_convention_still_reproduces_the_divergence():
    """The selectable fields keep the pre-fix statistics measurable."""
    filled = _training_nearfield_fill(
        torch.full((8, DEPTH_HEIGHT * DEPTH_WIDTH), 0.1)
    )
    model = _noise_model(num_envs=8, too_close_fill="max")

    far = 0
    total = 0
    for _ in range(20):
        noisy = model(filled.clone())
        far += int((noisy >= DEPTH_MAX - 1e-3).sum())
        total += noisy.numel()

    assert 0.45 < far / total < 0.55, (
        f"the retired convention should invert about half the class, "
        f"got {far / total:.4f}"
    )
