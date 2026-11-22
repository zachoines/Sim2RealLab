"""The training run's manifest: what a run directory trained on, written at run start.

``train_strafer_navigation.py`` writes ``<log_dir>/train_manifest.json`` before its first iteration. It records the
task ID, the code's git SHA and whether the tree had uncommitted changes, the seed, the env count, the iteration
budget, the command line, and the resume chain. The chain is the checkpoint a resumed run continued from, followed by
that run's own chain, so a multi-leg run names every leg back to its first. ``export_policy.py`` reads the manifest of
the run a checkpoint belongs to and records its task as the sidecar's training env.

Pure Python: importable without Isaac Sim.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "train_manifest.json"
SCHEMA_VERSION = 1


def _git(*args: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return result.stdout


def code_revision(path: str | Path | None = None) -> tuple[str | None, bool | None]:
    """The git SHA of the tree holding ``path`` (default: this module) and whether its tracked files differ from it.

    ``(None, None)`` where git or the repository is unavailable. Untracked files do not count as changes.
    """
    cwd = Path(path).resolve() if path is not None else Path(__file__).resolve().parent
    if cwd.is_file():
        cwd = cwd.parent
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if not sha or not sha.strip():
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    return sha.strip(), (None if status is None else bool(status.strip()))


def read_manifest_for_checkpoint(checkpoint: str | Path) -> dict | None:
    """The manifest of the run directory holding ``checkpoint``, or None where that run wrote none."""
    path = Path(checkpoint).expanduser().resolve().parent / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def resume_chain(resume: str | Path | None) -> list[dict]:
    """The legs a run continues, newest first: the resumed checkpoint and its run's task, then that run's own chain.

    A leg whose run wrote no manifest records its checkpoint and run directory with ``task_id`` None, and ends the
    chain: what it continued is not recorded.
    """
    if not resume:
        return []
    checkpoint = Path(resume).expanduser().resolve()
    parent = read_manifest_for_checkpoint(checkpoint)
    leg = {"checkpoint": str(checkpoint), "run_dir": str(checkpoint.parent),
           "task_id": parent.get("task_id") if parent else None}
    return [leg] + (list(parent.get("resume_chain", [])) if parent else [])


def build_train_manifest(*, task_id: str, seed: int, num_envs: int, max_iterations: int,
                         num_steps_per_env: int | None = None, resume: str | Path | None = None,
                         argv: list[str] | None = None, code_path: str | Path | None = None,
                         started_at: str | None = None) -> dict:
    """The manifest of one training run, as ``write_train_manifest`` stores it."""
    sha, dirty = code_revision(code_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "seed": int(seed),
        "num_envs": int(num_envs),
        "max_iterations": int(max_iterations),
        "num_steps_per_env": None if num_steps_per_env is None else int(num_steps_per_env),
        "resume": None if not resume else str(Path(resume).expanduser().resolve()),
        "resume_chain": resume_chain(resume),
        "argv": list(argv) if argv is not None else None,
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
    }


def write_train_manifest(log_dir: str | Path, manifest: dict) -> Path:
    """Write ``manifest`` to ``<log_dir>/train_manifest.json`` and return the path."""
    path = Path(log_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
