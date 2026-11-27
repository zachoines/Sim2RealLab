# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The training run's manifest: what it records, the resume chain back to a run's first leg, and the export reading it.

No Kit / GPU.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from strafer_shared.constants import POLICY_PERIOD_S
from strafer_lab.tools.train_manifest import (
    MANIFEST_NAME,
    build_train_manifest,
    code_revision,
    read_manifest_for_checkpoint,
    write_train_manifest,
)


def _git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "start"], check=True)
    (path / "tracked.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "file"],
                   check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _leg(tmp_path, name, task, resume=None):
    run = tmp_path / name
    manifest = build_train_manifest(task_id=task, seed=7, num_envs=52, max_iterations=1000, num_steps_per_env=48,
                                    resume=resume, argv=["train_strafer_navigation.py", "--env", task])
    write_train_manifest(run, manifest)
    (run / "model_999.pt").write_bytes(b"")
    return run / "model_999.pt"


def test_the_manifest_records_the_run_at_its_start(tmp_path):
    ckpt = _leg(tmp_path, "run_a", "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0")
    m = json.loads((ckpt.parent / MANIFEST_NAME).read_text())
    assert m["task_id"] == "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0"
    assert (m["seed"], m["num_envs"], m["max_iterations"], m["num_steps_per_env"]) == (7, 52, 1000, 48)
    assert m["resume"] is None and m["resume_chain"] == []
    assert m["argv"][-1] == "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0" and m["schema_version"] == 1
    assert {"git_sha", "git_dirty", "started_at"} <= set(m)


def test_the_resume_chain_names_every_leg_back_to_the_first(tmp_path):
    first = _leg(tmp_path, "run_a", "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0")
    second = _leg(tmp_path, "run_b", "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0", resume=first)
    third = _leg(tmp_path, "run_c", "Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0", resume=second)
    chain = read_manifest_for_checkpoint(third)["resume_chain"]
    assert [leg["run_dir"] for leg in chain] == [str(second.parent.resolve()), str(first.parent.resolve())]
    assert [leg["task_id"] for leg in chain] == ["Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0"] * 2


def test_a_leg_without_a_manifest_ends_the_chain_with_its_task_unrecorded(tmp_path):
    (tmp_path / "run_old").mkdir()
    old = tmp_path / "run_old" / "model_500.pt"
    old.write_bytes(b"")
    leg = _leg(tmp_path, "run_new", "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0", resume=old)
    chain = read_manifest_for_checkpoint(leg)["resume_chain"]
    assert chain == [{"checkpoint": str(old.resolve()), "run_dir": str(old.parent.resolve()), "task_id": None}]
    assert read_manifest_for_checkpoint(old) is None


def test_the_revision_is_the_trees_head_and_dirty_only_on_tracked_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _git_repo(repo)
    assert code_revision(repo) == (head, False)
    (repo / "untracked.txt").write_text("x\n")
    assert code_revision(repo) == (head, False)
    (repo / "tracked.txt").write_text("b\n")
    assert code_revision(repo) == (head, True)
    outside = tmp_path / "outside"
    outside.mkdir()
    assert code_revision(outside) == (None, None)


def test_the_export_sidecar_carries_the_training_env_beside_the_export_env(tmp_path):
    export_policy = pytest.importorskip("export_policy")
    ckpt = _leg(tmp_path, "run_a", "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0")
    common = dict(policy_variant="DEPTH_SUBGOAL", obs_dim=3619, action_dim=3,
                  env_id="Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0", training_preset="P", source_checkpoint=ckpt,
                  formats=["pt"], is_recurrent=True, trained_period_s=POLICY_PERIOD_S, git_commit="x",
                  export_timestamp="t")
    sidecar = export_policy.write_metadata_sidecar(tmp_path / "out" / "policy",
                                                   training_env_id="Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0",
                                                   training_manifest=ckpt.parent / MANIFEST_NAME, **common)
    payload = json.loads(sidecar.read_text())
    assert payload["env_id"] == "Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0"
    assert payload["training_env_id"] == "Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0"
    assert payload["training_manifest"] == MANIFEST_NAME  # the file beside the checkpoint, no host path
    bare = json.loads(export_policy.write_metadata_sidecar(tmp_path / "out" / "bare", **common).read_text())
    assert bare["training_env_id"] is None and bare["training_manifest"] is None  # absent, and said so
