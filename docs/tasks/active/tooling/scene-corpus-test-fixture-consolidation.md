# Consolidate the scene-corpus test fixtures and restore the ceiling sweep

**Type:** refactor
**Owner:** DGX agent
**Priority:** P3
**Estimate:** S–M
**Branch:** task/scene-corpus-test-fixture-consolidation

## Story

As **the strafer_lab test suite**, I want **one way to author a stand-in scene
corpus**, so that **a test needing scene geometry stops choosing between three
helpers with two incompatible layouts, and the sweeps that were narrowed to dodge
the corpus dependency can be widened back.**

## Context bundle

- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/conventions.md](../../context/conventions.md)
- [context/multi-room-architecture.md](../../context/multi-room-architecture.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)

## Context

PR #204 made the env cfg tests hermetic by adding
`test_sim/common/scenes.py`, which authors a stand-in corpus and repoints
`SCENE_USD_DIR`. It deliberately did not touch the copies that already existed,
so there are now three helpers authoring a scene corpus across two layouts:

| helper | layout |
|---|---|
| `test_sim/common/scenes.py::write_stub_scene_corpus` | `<scenes>/<stem>/export/<stem>.usdc` + top-level symlink |
| `tests/harness/test_lightest_scene.py::_make_scene` | same |
| `tests/harness/test_scene_paths.py` fixture | same |
| `tests/harness/test_bridge_spawn_from_occupancy.py::_author_scene` | `<scenes>/<stem>/<stem>.usdc`, **no** `export/`, **no** symlink, plus a real occupancy sidecar and USD room footprints |

The last disagrees with the other three. Both layouts are legitimate — `scene_dir_for`
resolves either, and the occupancy-bearing one is what a spawn-derivation test
needs — but a contributor writing a fourth test has no way to tell which is the
real corpus shape, and the divergence hides which behaviours actually depend on
the `export/` level and the symlink.

Separately, the corpus dependency narrowed a real check.
[`tests/navigation/test_ceiling_surface_cfg.py:110-126`](../../../../source/strafer_lab/tests/navigation/test_ceiling_surface_cfg.py)
sweeps the composed variants asserting that face culling is enabled exactly where
a ceiling entity exists — and filters to `procroom` with the comment "the
Infinigen variants bind a generated scene USD that a checkout need not carry".
The capture and bridge variants are therefore permanently outside a check whose
whole value is that it holds on *every* variant. `hermetic_scene_corpus` is
exactly what that filter was missing, so the filter can go and the assertion can
cover what it was written to cover.

Note that `tests/navigation/` is the pure suite, not `test_sim/` — confirm the
import resolves there before relying on it.

`test_sim/common/scenes.py` also carries `scene_source_kind()`, duplicating the
private `_scene_source_kind` in the ceiling test; that duplication should
collapse rather than persist.

## Acceptance criteria

- [ ] One documented helper authors a stand-in scene corpus, with the occupancy
      sidecar and room footprints as an option rather than a separate layout. If
      the two layouts must stay distinct, the reason is written down where a
      contributor choosing between them will read it.
- [ ] The `procroom`-only filter in `test_culling_is_set_exactly_where_the_ceiling_is`
      is removed and the sweep covers every `StraferNavCfg_*`, with the ceiling /
      culling equivalence still asserted.
- [ ] `_scene_source_kind` in the ceiling test is replaced by the shared
      `scene_source_kind`, or the shared one is removed in favour of the local —
      not both.
- [ ] All 26 frozen goldens in `test_sim/env/test_composition_contract.py`
      recompute byte-identical, stated in the PR.
- [ ] The full pure suite and the contract two-file command pass in a git
      worktree with no `Assets/generated`, at their current counts.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit.

## Investigation pointers

- `source/strafer_lab/test_sim/common/scenes.py` — the shared helper and fixture.
- `source/strafer_lab/tests/navigation/test_ceiling_surface_cfg.py:110-126` — the
  narrowed sweep and the comment naming the reason.
- `source/strafer_lab/tests/harness/test_bridge_spawn_from_occupancy.py:29-60` —
  the occupancy-bearing layout.
- `source/strafer_lab/tests/harness/test_lightest_scene.py:25-57` — the
  symlink-bearing layout the shared helper mirrors.
- `source/strafer_lab/strafer_lab/tools/scene_connectivity.py:176` —
  `scene_dir_for`, which resolves both layouts and is why they diverged unnoticed.

## Out of scope

- The `models/` artifact-discovery case — see
  [`deployed-artifact-test-discovery`](deployed-artifact-test-discovery.md).
- Any production change to `_get_scene_usd_paths` or `SCENE_USD_DIR`.
- Re-freezing any golden.
