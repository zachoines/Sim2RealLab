"""Schema-conformance tests for the bridge-mode gym-side obs dump.

The dumper is the training half of the train<->deploy observation-parity join
(the deployment half lives in the inference node). These tests pin the
record/assembly contract — field ordering against ``PolicyVariant``, the
computed-vs-NaN dimension split, per-field scaling, and JSONL schema shape —
without a live Kit process: they exercise the pure functions and never call the
Kit-bound term evaluator (that is covered by the rig smoke).

The strict numeric join against a node dump is exercised in the
``strafer_inference`` parity suite; here we only prove the producer emits what
that consumer contracts for.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from strafer_lab.bridge import obs_dump
from strafer_lab.bridge.obs_dump_terms import make_bridge_obs_dumper
from strafer_shared.policy_interface import PolicyVariant, assemble_observation

_ALL_VARIANTS = (
    PolicyVariant.NOCAM,
    PolicyVariant.DEPTH,
    PolicyVariant.NOCAM_SUBGOAL,
    PolicyVariant.DEPTH_SUBGOAL,
)


def _computable_term_values(variant: PolicyVariant) -> dict[str, np.ndarray]:
    """Distinct, non-trivial raw values for each computable field of a variant.

    Values are unique per dim so a mis-order would surface as a value mismatch,
    not just a shape one.
    """
    values: dict[str, np.ndarray] = {}
    base = 1.0
    for field in variant.fields:
        if field.key in obs_dump._COMPUTABLE_KEYS:
            values[field.key] = base + np.arange(field.dims, dtype=np.float32)
            base += 100.0
    return values


def _field_ranges(variant: PolicyVariant):
    """(computable_idx, nan_idx) flat index arrays, walked from PolicyVariant.

    No dim literals: the split follows the module's own ``_COMPUTABLE_KEYS`` so
    the test tracks the contract, not a hardcoded layout.
    """
    computable: list[int] = []
    nan: list[int] = []
    off = 0
    for field in variant.fields:
        target = computable if field.key in obs_dump._COMPUTABLE_KEYS else nan
        target.extend(range(off, off + field.dims))
        off += field.dims
    return np.asarray(computable, dtype=int), np.asarray(nan, dtype=int)


class TestVectorAssembly:
    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_length_matches_obs_dim(self, variant):
        vec = obs_dump.build_obs_vector(variant, _computable_term_values(variant))
        assert vec.shape == (variant.obs_dim,)
        assert vec.dtype == np.float32

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_nan_policy_is_exactly_the_non_computable_dims(self, variant):
        vec = obs_dump.build_obs_vector(variant, _computable_term_values(variant))
        computable_idx, nan_idx = _field_ranges(variant)
        assert np.isfinite(vec[computable_idx]).all()
        assert np.isnan(vec[nan_idx]).all()
        # The two blocks partition the whole vector — nothing unaccounted for.
        assert computable_idx.size + nan_idx.size == variant.obs_dim

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_computed_dims_are_raw_times_field_scale(self, variant):
        terms = _computable_term_values(variant)
        vec = obs_dump.build_obs_vector(variant, terms)
        off = 0
        for field in variant.fields:
            if field.key in obs_dump._COMPUTABLE_KEYS:
                expected = terms[field.key] * field.scale
                np.testing.assert_allclose(
                    vec[off : off + field.dims], expected, rtol=0, atol=0
                )
            off += field.dims

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_assembly_equals_assemble_observation_of_the_raw_dict(self, variant):
        # The dumper must delegate scaling/ordering to the deploy assembler, so
        # the computed dims match the node's assembled vector bit-for-bit.
        terms = _computable_term_values(variant)
        raw = obs_dump.build_raw_term_dict(variant, terms)
        expected = assemble_observation(raw, variant)
        vec = obs_dump.build_obs_vector(variant, terms)
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(vec[finite], expected[finite])
        assert np.isnan(vec[~finite]).all()

    def test_depth_block_only_for_depth_variants(self):
        depth_keys = {"depth_image"}
        for v in (PolicyVariant.DEPTH, PolicyVariant.DEPTH_SUBGOAL):
            assert depth_keys <= {f.key for f in v.fields}
            vec = obs_dump.build_obs_vector(v, _computable_term_values(v))
            # 3600 finite depth dims — the depth term is computable, not NaN.
            assert np.isfinite(vec).sum() >= 3600
        for v in (PolicyVariant.NOCAM, PolicyVariant.NOCAM_SUBGOAL):
            assert depth_keys.isdisjoint({f.key for f in v.fields})

    def test_missing_computable_term_raises(self):
        terms = _computable_term_values(PolicyVariant.NOCAM_SUBGOAL)
        del terms["imu_gyro"]
        with pytest.raises(KeyError):
            obs_dump.build_obs_vector(PolicyVariant.NOCAM_SUBGOAL, terms)

    def test_wrong_dim_term_raises(self):
        terms = _computable_term_values(PolicyVariant.NOCAM_SUBGOAL)
        terms["imu_accel"] = np.zeros(2, dtype=np.float32)  # should be 3
        with pytest.raises(ValueError):
            obs_dump.build_obs_vector(PolicyVariant.NOCAM_SUBGOAL, terms)


class TestRecordSchema:
    def test_record_fields_types_and_json_roundtrip(self):
        variant = PolicyVariant.NOCAM_SUBGOAL
        vec = obs_dump.build_obs_vector(variant, _computable_term_values(variant))
        rec = obs_dump.obs_record(variant, 12.5, vec)

        assert set(rec) == {"t_sim", "variant", "obs", "referent"}
        assert isinstance(rec["t_sim"], float) and rec["t_sim"] == 12.5
        assert rec["variant"] == "NOCAM_SUBGOAL"
        assert rec["referent"] is None
        assert isinstance(rec["obs"], list) and len(rec["obs"]) == variant.obs_dim

        # One JSON object per line, and NaN dims survive the round-trip as NaN
        # (json emits them as the literal NaN token, which json.loads reads back).
        line = json.dumps(rec)
        assert "\n" not in line
        back = json.loads(line)
        assert back["variant"] == "NOCAM_SUBGOAL"
        assert len(back["obs"]) == variant.obs_dim
        assert any(math.isnan(x) for x in back["obs"])

    def test_make_dumper_disabled_on_empty_path(self):
        # The factory lives in the Kit-side module but imports Kit-free and
        # returns None before touching Isaac Lab, so the guard is testable here.
        assert make_bridge_obs_dumper("", "DEPTH_SUBGOAL") is None
        assert make_bridge_obs_dumper("   ", "DEPTH_SUBGOAL") is None

    def test_writer_writes_one_line_per_write(self, tmp_path):
        # The pure writer takes already-evaluated term values, so it exercises
        # the serialization path without Kit (real term evaluation is pinned by
        # the Kit test in test_sim/bridge/).
        out = tmp_path / "gym_obs.jsonl"
        variant = PolicyVariant.NOCAM_SUBGOAL
        terms = _computable_term_values(variant)
        with obs_dump.ObsDumpWriter(str(out), variant) as writer:
            writer.write(0.0, terms)
            writer.write(1.0 / 30.0, terms)

        lines = out.read_text().splitlines()
        assert len(lines) == 2
        recs = [json.loads(x) for x in lines]
        assert [r["t_sim"] for r in recs] == [0.0, 1.0 / 30.0]
        assert all(len(r["obs"]) == variant.obs_dim for r in recs)
        assert all(r["variant"] == "NOCAM_SUBGOAL" for r in recs)
        assert all(r["referent"] is None for r in recs)
