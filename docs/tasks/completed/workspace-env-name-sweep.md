# Sweep workspace-environment names out of source

**Type:** docs / cleanup
**Owner:** DGX
**Priority:** P3 — convention enforcement; not gating.
**Estimate:** S
**Branch:** `task/workspace-env-name-sweep`
**Shipped:** PR #94 @ `3d1e4e5` — `rg -n "env_isaaclab3|env_infinigen|\.venv_vlm" source` clean; all changed files `py_compile` clean.

## Story

As a contributor I want source files to describe their environment by its
**requirement**, not by my workspace's conda/venv names, so the code
doesn't rot when an environment is renamed or a second contributor uses a
different one.

## Motivation

[`scene-metadata-in-usd`](scene-metadata-in-usd.md) (PR #90)
added the **Workspace environment names** rule to
[`conventions.md`](../context/conventions.md#workspace-environment-names)
— source must not name `.venv_vlm` / `env_isaaclab3` / `env_infinigen`. #90
fixed the files it touched; ~11 pre-existing mentions remained in files it
didn't. This sweep clears them so the convention holds repo-wide.

## Acceptance

- [ ] No `.venv_vlm` / `env_isaaclab3` / `env_infinigen` in `source/**/*.py`
  (docstrings / comments / CLI help) — `rg -n "env_isaaclab3|env_infinigen|\.venv_vlm" source` is clean.
- [ ] Each mention reworded to its requirement (`pxr` / no-`pxr` / an Isaac
  Sim Kit env) per the convention.
- [ ] Docstring/comment-only — no behavior change.

## Out of scope

- The env ↔ name source-of-truth surfaces (`repo-topology.md`,
  `env_setup.sh`, `.env.example`, `Makefile`, package READMEs) keep their
  names — they own the mapping.
- Exposing `.venv_vlm` as a `STRAFER_VLM_PYTHON` export (a separate
  operator-tooling brief if wanted).

## Triggered by

PR #90 review — env names in source were flagged as a smell; the convention
was added there, and this sweep applies it to the remaining files.
