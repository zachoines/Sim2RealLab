#!/usr/bin/env python3
"""Dump the config-hash goldens and their full canonical JSON preimages.

The composition-contract test stores 26 hashes as in-file constants. A hash
says only *that* something moved; the preimage says *what*. Dumping the
preimages before a sim-stack bump turns the post-bump comparison into a
textual diff that names every changed field, and the named field set is the
input to the drop-subtraction attribution the contract test already supports.

Serialization is not reimplemented here: the test module is imported by path
and its own ``_canon`` / ``_contract`` / ``_hash`` are called, so a preimage
is byte-identical to the one that produced the stored golden.

Usage (no Kit boot; ~4 s):

    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
    $STRAFER_ISAACLAB_PYTHON \
      docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/dump_golden_preimages.py \
      --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/goldens

Writes ``<out>/preimages/*.json`` (one per hashed object) and
``<out>/golden-hashes.json`` (stored vs recomputed, with a match verdict).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
TEST_MODULE = REPO / "source/strafer_lab/test_sim/env/test_composition_contract.py"


def load_contract_module():
    """Import the contract test by path, without pytest or its conftest."""
    spec = importlib.util.spec_from_file_location("_contract_mod", TEST_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_contract_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def canon_json(obj) -> str:
    """The exact byte sequence the golden hash is taken over, pretty-printed
    for diffing. The hash itself is recomputed from the compact form."""
    return json.dumps(obj, sort_keys=True, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    m = load_contract_module()
    pre_dir = args.out / "preimages"
    pre_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    def record(kind: str, name: str, preimage, stored: str | None):
        got = m._hash(preimage)
        (pre_dir / f"{kind}-{name}.json").write_text(canon_json(preimage) + "\n")
        records.append(
            {
                "kind": kind,
                "name": name,
                "stored_golden": stored,
                "recomputed": got,
                "match": (stored is None) or (got == stored),
                "preimage_file": f"preimages/{kind}-{name}.json",
            }
        )

    # 22 contract hashes — the full composed-variant contract.
    for name in sorted(m._COMPOSED_RL):
        cfg = m._COMPOSED_RL[name]()
        record("contract", name, m._contract(cfg), m._CONTRACT_GOLDENS[name])

    # 1 depth-obs golden — both observation groups of the reference depth variant.
    record(
        "depthobs",
        "RLDepth_Real",
        m._canon(m.composed.StraferNavCfg_RLDepth_Real().observations),
        m._DEPTH_OBS_GOLDEN,
    )

    # 2 policy-obs layout goldens — the deployment-contract half, noise dropped.
    # Emitted once per profile from the first variant carrying it; the test's
    # tier-invariance sweep is what proves one layout per profile across all 22.
    seen: set[str] = set()
    for name in sorted(m._COMPOSED_RL):
        cfg = m._COMPOSED_RL[name]()
        profile = m._obs_profile(cfg)
        if profile in seen:
            continue
        seen.add(profile)
        record(
            "layout",
            profile,
            m._canon(cfg.observations.policy, drop=frozenset({"noise"})),
            m._POLICY_OBS_LAYOUT_GOLDENS[profile],
        )

    # 1 palette golden — the rigid-object palette no contract hash can see.
    record(
        "palette",
        "pre_enrichment",
        m._palette_sig(m._COMPOSED_RL["RLNoCam"]().scene.room_primitives),
        m._PALETTE_GOLDEN,
    )

    # Per-variant layout attribution: which profile each variant resolves to,
    # so a post-bump profile flip is readable without recomputing every cfg.
    profiles = {
        name: m._obs_profile(m._COMPOSED_RL[name]()) for name in sorted(m._COMPOSED_RL)
    }

    summary = {
        "n_records": len(records),
        "n_matching": sum(1 for r in records if r["match"]),
        "all_match": all(r["match"] for r in records),
        "variant_obs_profile": profiles,
        "records": records,
    }
    (args.out / "golden-hashes.json").write_text(json.dumps(summary, indent=2) + "\n")

    for r in records:
        flag = "ok " if r["match"] else "MISMATCH"
        print(f"{flag} {r['kind']:9s} {r['name']:34s} {r['recomputed']}")
    print(f"\n{summary['n_matching']}/{summary['n_records']} recomputed == stored golden")
    return 0 if summary["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
