# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Kit test: the bridge obs dump equals training's assembled obs on the
computable dims, evaluated against a live env.

Pins the term-evaluation wiring the Kit-free pure suite cannot reach — which
sensor cfg names, which term functions, ordering, and scales the evaluator uses.
It brings up a depth env, writes dump lines through the real evaluator
(``strafer_lab.bridge.obs_dump_terms``), and asserts each line's computable dims
match Isaac Lab's own ObservationManager output at the same positions, with the
referent-derived and ``last_action`` dims NaN exactly where the variant declares
them. Because the evaluator calls the term functions verbatim, this is a wiring
test: a wrong sensor name, term, order, or scale shifts the compared dims and
fails here.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/bridge/test_obs_dump_terms.py -v
"""

import json

import numpy as np
import pytest
import torch

from strafer_shared.policy_interface import PolicyVariant

from strafer_lab.bridge import obs_dump
from strafer_lab.bridge.obs_dump_terms import evaluate_obs_terms, make_bridge_obs_dumper


def _step(env, n: int = 1):
    """Step the env ``n`` times with a zero action; return env 0's policy obs."""
    action = torch.zeros(env.num_envs, 3, device=env.device)
    obs = None
    for _ in range(n):
        obs, *_ = env.step(action)
    return obs["policy"][0].detach().cpu().numpy()


def _index_blocks(variant: PolicyVariant):
    """(scalar_computable, depth, nan) flat index arrays, walked from the variant.

    No dim literals: the split follows the pure module's ``COMPUTABLE_KEYS`` and
    the depth field, so the test tracks the contract, not a hardcoded layout.
    """
    scalar: list[int] = []
    depth: list[int] = []
    nan: list[int] = []
    off = 0
    for f in variant.fields:
        rng = range(off, off + f.dims)
        if f.key == "depth_image":
            depth.extend(rng)
        elif f.key in obs_dump.COMPUTABLE_KEYS:
            scalar.extend(rng)
        else:
            nan.extend(rng)
        off += f.dims
    return np.asarray(scalar, int), np.asarray(depth, int), np.asarray(nan, int)


class TestEvaluatorContract:
    def test_evaluator_keys_match_the_computable_contract(self, depth_env):
        depth_env.reset()
        _step(depth_env, 5)
        for v in (PolicyVariant.DEPTH_SUBGOAL, PolicyVariant.NOCAM_SUBGOAL):
            terms = evaluate_obs_terms(depth_env, v)
            expected = {f.key for f in v.fields if f.key in obs_dump.COMPUTABLE_KEYS}
            assert set(terms) == expected
            for key, arr in terms.items():
                assert np.isfinite(arr).all(), f"{key} has non-finite values"


class TestDumpMatchesEnvObs:
    @pytest.mark.parametrize(
        "variant", [PolicyVariant.NOCAM_SUBGOAL, PolicyVariant.DEPTH_SUBGOAL]
    )
    def test_dumped_line_matches_env_manager_on_computable_dims(
        self, depth_env, tmp_path, variant
    ):
        depth_env.reset()
        # The env's own clean assembled policy obs (env 0). No step happens
        # between here and the dump, so both read the same post-step buffers.
        env_obs = _step(depth_env, 5)

        out = tmp_path / f"gym_{variant.name}.jsonl"
        dumper = make_bridge_obs_dumper(str(out), variant.name)
        assert dumper is not None
        dumper.write(depth_env, t_sim=0.0)
        dumper.close()

        rec = json.loads(out.read_text().splitlines()[0])
        assert rec["variant"] == variant.name
        assert rec["referent"] is None
        dumped = np.asarray(rec["obs"], dtype=np.float32)
        assert dumped.shape == (variant.obs_dim,)

        scalar_idx, depth_idx, nan_idx = _index_blocks(variant)
        # NaN policy: exactly the referent-derived + last_action dims.
        assert np.isnan(dumped[nan_idx]).all()
        assert np.isfinite(dumped[scalar_idx]).all()
        # Wiring pin: the computable scalar dims equal Isaac Lab's own assembled
        # obs at the same positions. DEPTH/NOCAM and their subgoal variants share
        # these positions; the env runs the Ideal profile, so there is no noise.
        np.testing.assert_allclose(
            dumped[scalar_idx], env_obs[scalar_idx], rtol=1e-4, atol=1e-4
        )
        if depth_idx.size:
            assert np.isfinite(dumped[depth_idx]).all()
            assert (dumped[depth_idx] >= 0.0).all()
            assert (dumped[depth_idx] <= 1.0).all()
            np.testing.assert_allclose(
                dumped[depth_idx], env_obs[depth_idx], rtol=1e-3, atol=1e-3
            )

    def test_multiple_lines_are_written_and_valid(self, depth_env, tmp_path):
        v = PolicyVariant.DEPTH_SUBGOAL
        depth_env.reset()
        _step(depth_env, 5)

        out = tmp_path / "gym.jsonl"
        dumper = make_bridge_obs_dumper(str(out), v.name)
        action = torch.zeros(depth_env.num_envs, 3, device=depth_env.device)
        for i in range(3):
            depth_env.step(action)
            dumper.write(depth_env, t_sim=i / 30.0)
        dumper.close()

        lines = out.read_text().splitlines()
        assert len(lines) == 3
        for i, ln in enumerate(lines):
            rec = json.loads(ln)
            assert rec["variant"] == "DEPTH_SUBGOAL"
            assert rec["referent"] is None
            assert len(rec["obs"]) == v.obs_dim
            assert rec["t_sim"] == pytest.approx(i / 30.0)
