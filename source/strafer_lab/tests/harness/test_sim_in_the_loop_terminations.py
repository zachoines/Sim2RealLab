"""Sim-in-the-loop runs one continuous episode: all env terminations nulled.

Bridge and harness mode must behave like the real robot for a mission's
duration — a collision or flip is a bump, not an env reset that teleports the
robot to a fresh spawn and re-chases until timeout. The autonomy executor owns
mission end, so the override nulls *every* termination term (enumerating, not a
fixed name list, so composed ProcRoom / Subgoal variants are covered too). The
training cfg classes are untouched; only this runtime override changes.

Hermetic: no Kit boot, no live env. ``run_sim_in_the_loop`` is imported inside
each test (it pulls ``isaaclab.app`` at module top) and driven against
``SimpleNamespace`` cfg stubs — the same in-script pattern
``test_bridge_spawn_from_occupancy.py`` uses for the ``--scene-usd`` override.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _import_rsil():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_sim_in_the_loop as rsil

    return rsil


def _stub_env_cfg(terminations):
    """A minimal env cfg the override can walk without an Isaac Lab env.

    ``reset_robot=None`` skips the yaw-pin branch; matched decimation and
    ``render_interval=None`` (below) skip those branches, isolating the
    termination-disable path.
    """
    return SimpleNamespace(
        events=SimpleNamespace(reset_robot=None),
        terminations=terminations,
        decimation=1,
        sim=SimpleNamespace(render_interval=1),
    )


class TestDisableEnvTerminations:
    def test_all_active_terms_nulled_and_named(self):
        rsil = _import_rsil()
        # A Subgoal-composed set: base + ProcRoom + Subgoal terms all present.
        terms = SimpleNamespace(
            time_out=object(),
            robot_flipped=object(),
            sustained_collision=object(),
            goal_reached=object(),
            path_complete=object(),
            off_path_divergence=object(),
        )
        disabled = rsil._disable_env_terminations(terms)

        assert set(disabled) == {
            "time_out",
            "robot_flipped",
            "sustained_collision",
            "goal_reached",
            "path_complete",
            "off_path_divergence",
        }
        for name in disabled:
            assert getattr(terms, name) is None

    def test_already_none_terms_untouched_and_not_reported(self):
        rsil = _import_rsil()
        terms = SimpleNamespace(
            time_out=object(),
            robot_flipped=None,  # already off — must not be re-reported
            sustained_collision=object(),
        )
        disabled = rsil._disable_env_terminations(terms)

        assert set(disabled) == {"time_out", "sustained_collision"}
        assert "robot_flipped" not in disabled
        assert terms.robot_flipped is None
        assert terms.time_out is None
        assert terms.sustained_collision is None


class TestApplyOverridesTerminationGate:
    def test_disable_path_nulls_all_terms_and_prints(self, capsys):
        rsil = _import_rsil()
        terms = SimpleNamespace(time_out=object(), sustained_collision=object())
        cfg = _stub_env_cfg(terms)

        rsil._apply_sim_in_the_loop_overrides(
            cfg,
            pin_yaw=0.0,
            disable_terminations=True,
            decimation=1,
            render_interval=None,
        )

        assert cfg.terminations.time_out is None
        assert cfg.terminations.sustained_collision is None
        out = capsys.readouterr().out
        assert "env terminations disabled:" in out

    def test_keep_flag_leaves_terminations_unmutated(self):
        rsil = _import_rsil()
        time_out, collision = object(), object()
        terms = SimpleNamespace(time_out=time_out, sustained_collision=collision)
        cfg = _stub_env_cfg(terms)

        rsil._apply_sim_in_the_loop_overrides(
            cfg,
            pin_yaw=0.0,
            disable_terminations=False,
            decimation=1,
            render_interval=None,
        )

        assert cfg.terminations.time_out is time_out
        assert cfg.terminations.sustained_collision is collision
