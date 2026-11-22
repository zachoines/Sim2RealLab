# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""A failed sensor repeats its previous reading, as a held source does in the deployed node's cache.

The IMU and encoder models, built from the composed robust cfg with their stuck rate raised, must re-emit the previous
emitted reading bit for bit, never zeros; a reset forgets the previous reading, so an env's first step after one
passes the current reading. The depth model has no failure mode: a failed frame is a dropped one, which repeats the
previous frame and never reads all far. The composed cfgs carry the stated rates and none of the removed fields. The
real tier's IMU and encoder models are pinned bit for bit, so an added random draw on that tier fails. (On the
robot the IMU's two channels hold together, where training draws them apart; the encoders and the body velocity are
one reading in both.)

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

import pytest
import torch

N = 32
ACC, GYR, ENC, DEPTH = "imu_linear_acceleration", "imu_angular_velocity", "wheel_encoder_velocities", "depth_image"


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(1)


@functools.lru_cache(maxsize=None)
def _cfg(tier):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, f"StraferNavCfg_RLDepthSubgoal_{tier}")()


def _noise(tier, term):
    return getattr(_cfg(tier).observations.policy, term).noise


def _model(tier, term, **overrides):
    cfg = _noise(tier, term).copy()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg.class_type(cfg, N, "cpu")


def _reading(term, gen):
    width = {ACC: 3, GYR: 3, ENC: 4}[term]
    return (torch.rand(N, width, generator=gen) - 0.5) * (2000.0 if term == ENC else 20.0) + 1.0


@pytest.mark.parametrize("term", [ACC, GYR, ENC])
def test_a_stuck_source_repeats_its_previous_reading_bit_for_bit(term):
    model = _model("Robust", term, stuck_probability=1.0, latency_steps=0)
    gen = torch.Generator().manual_seed(5)
    first = model(_reading(term, gen))
    for _ in range(10):
        out = model(_reading(term, gen))
        assert torch.equal(out, first)
        assert not bool((out == 0.0).all(dim=-1).any()), "a stuck source must not read zeros"


@pytest.mark.parametrize("term", [ACC, GYR, ENC])
def test_a_reset_forgets_the_previous_reading(term):
    model = _model("Robust", term, stuck_probability=1.0, latency_steps=0)
    gen = torch.Generator().manual_seed(6)
    first = model(_reading(term, gen))
    model.reset(torch.tensor([3, 17]))
    data = _reading(term, gen)
    out = model(data.clone())
    kept = [e for e in range(N) if e not in (3, 17)]
    assert torch.equal(out[kept], first[kept])
    assert not torch.equal(out[[3, 17]], first[[3, 17]])  # the reset envs pass their current reading
    assert not bool((out[[3, 17]] == 0.0).all())


@pytest.mark.parametrize("term", [ACC, ENC])
def test_the_draw_is_per_env_and_holds_the_whole_reading(term):
    # Quantization off: a fresh encoder reading rounded to whole ticks can match the previous one in a component by
    # chance, which would read as a partly held row.
    extra = {"enable_quantization": False} if term == ENC else {}
    model = _model("Robust", term, stuck_probability=0.5, latency_steps=0, **extra)
    gen = torch.Generator().manual_seed(7)
    prev = model(_reading(term, gen))
    stuck_rows = fresh_rows = 0
    for _ in range(20):
        out = model(_reading(term, gen))
        same = out == prev
        assert bool((same.all(dim=-1) | ~same.any(dim=-1)).all()), "a row is held whole or not at all"
        stuck_rows += int(same.all(dim=-1).sum())
        fresh_rows += int((~same.any(dim=-1)).sum())
        prev = out
    assert stuck_rows and fresh_rows


def test_a_failed_depth_frame_repeats_the_previous_one_and_never_reads_far():
    cfg = _noise("Robust", DEPTH)
    assert not hasattr(cfg, "failure_probability")
    model = _model("Robust", DEPTH, frame_drop_prob=1.0, hold_fraction_range=(0.0, 0.0), latency_steps=0,
                   latency_steps_range=None)
    gen = torch.Generator().manual_seed(8)
    first = model(1.0 + 3.0 * torch.rand(N, cfg.height * cfg.width, generator=gen))
    for _ in range(10):
        out = model(1.0 + 3.0 * torch.rand(N, cfg.height * cfg.width, generator=gen))
        assert torch.equal(out, first)
        assert not bool((out == cfg.max_range).all(dim=-1).any())


def test_the_composed_rates_are_as_stated():
    """Robust: each IMU term stuck at 2e-4 (its 1e-4 stuck and 1e-4 failed, both now repeats), the encoders at 1e-4,
    the depth frame drop at 0.011 (0.01 and the 0.001 of a failed frame). Real: none."""
    for term in (ACC, GYR):
        assert _noise("Robust", term).stuck_probability == pytest.approx(2e-4)
        assert _noise("Real", term).stuck_probability == 0.0
    assert _noise("Robust", ENC).stuck_probability == pytest.approx(1e-4)
    assert _noise("Real", ENC).stuck_probability == 0.0
    assert _noise("Robust", DEPTH).frame_drop_prob == pytest.approx(0.011)
    assert _noise("Real", DEPTH).frame_drop_prob == pytest.approx(0.001)


def test_the_removed_fields_are_gone():
    from strafer_lab.tasks.navigation import sim_real_cfg
    from strafer_lab.tasks.navigation.mdp import noise_models

    for cls in (noise_models.IMUNoiseModelCfg, noise_models.EncoderNoiseModelCfg, noise_models.DepthNoiseModelCfg):
        assert "failure_probability" not in {f.name for f in dataclasses.fields(cls)}, cls.__name__
    failures = {f.name for f in dataclasses.fields(sim_real_cfg.SensorFailureCfg)}
    assert not failures & {"imu_failure_probability", "encoder_failure_probability"}
    assert {"imu_stuck_probability", "encoder_stuck_probability"} <= failures
    assert "hole_cluster_size" not in {f.name for f in dataclasses.fields(sim_real_cfg.DepthCameraNoiseCfg)}


# The real tier's models over 40 steps with resets, and the RNG state after. The IMU's are main's; the encoders' are
# the model without its tick errors, which the shared wheel reading removed.
# One literal per CPU normal-fill kernel family: the AVX2/AVX-512 kernel and the
# scalar one round a few float32 draws an ulp apart. One x86 machine re-freezes
# both: "avx2" run natively, "scalar" under ATEN_CPU_CAPABILITY=default.
REAL_FINGERPRINTS = {
    (ACC, "scalar"): "db54579037b4b7dfd08f6e202731d1e7d02deaa1937468dba5ffec4f0a5e81fe",
    (ACC, "avx2"): "bedd48d007eace53f40b0173e7317c8104fded739f60d4050d1b3c20c140494e",
    (GYR, "scalar"): "ebaf4fd31b20ea714c3a70885303f4c746e4ebecbb6f459975022f8098c3d4c3",
    (GYR, "avx2"): "ebaf4fd31b20ea714c3a70885303f4c746e4ebecbb6f459975022f8098c3d4c3",
    (ENC, "scalar"): "2080ec6f506f789d4f49aac076dd3e0936d78a8b9369aa56a78ce03e6bd4d1a4",
    (ENC, "avx2"): "2080ec6f506f789d4f49aac076dd3e0936d78a8b9369aa56a78ce03e6bd4d1a4",
}


def _normal_fill_family() -> str:
    """The CPU normal-fill kernel family this process dispatches to."""
    return "avx2" if torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512") else "scalar"


@pytest.mark.parametrize("term", [ACC, GYR, ENC])
def test_the_real_tier_is_unchanged(term):
    """The real tier takes no failure draw, so its noise is pinned bit for bit; any added draw moves this."""
    ncfg = _noise("Real", term)
    torch.manual_seed(31)
    model = ncfg.class_type(ncfg, 16, "cpu")
    gen = torch.Generator().manual_seed(32)
    torch.manual_seed(33)
    width = {ACC: 3, GYR: 3, ENC: 4}[term]
    rows = []
    for k in range(40):
        if k in (10, 25):
            model.reset(torch.tensor([2, 7, 11]))
        rows.append(model(0.5 + 4.0 * torch.rand(16, width, generator=gen)))
    rows.append(torch.rand(4))
    digest = hashlib.sha256(torch.cat([r.flatten() for r in rows]).numpy().tobytes()).hexdigest()
    assert digest == REAL_FINGERPRINTS[term, _normal_fill_family()]
