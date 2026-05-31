"""Unit tests for the teleop perf-knob resolvers (decimation + viewport).

Pure stdlib — runs in ``.venv_harness`` with no Isaac Sim.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.teleop_perf_knobs import (
    resolve_decimation,
    resolve_viewport_resolution,
)


class TestResolveDecimation:
    def test_none_keeps_defaults(self):
        # Nav defaults: sim.dt = 1/120, decimation = 4 → 30 Hz control rate.
        decim, dt = resolve_decimation(
            None, default_decimation=4, default_sim_dt=1.0 / 120.0,
        )
        assert decim == 4
        assert dt == pytest.approx(1.0 / 120.0)

    def test_halving_decimation_holds_control_rate(self):
        # 4 → 2 substeps must keep env_step rate at 30 Hz by doubling dt.
        decim, dt = resolve_decimation(
            2, default_decimation=4, default_sim_dt=1.0 / 120.0,
        )
        assert decim == 2
        assert dt == pytest.approx(1.0 / 60.0)
        # Control rate invariant: decimation * dt == default product.
        assert decim * dt == pytest.approx(4 * (1.0 / 120.0))
        # That product is 1/30 s → 30 Hz, unchanged.
        assert 1.0 / (decim * dt) == pytest.approx(30.0)

    def test_raising_decimation_also_holds_control_rate(self):
        decim, dt = resolve_decimation(
            8, default_decimation=4, default_sim_dt=1.0 / 120.0,
        )
        assert decim == 8
        assert 1.0 / (decim * dt) == pytest.approx(30.0)

    def test_non_positive_raises(self):
        for bad in (0, -1, -4):
            with pytest.raises(ValueError, match="positive"):
                resolve_decimation(
                    bad, default_decimation=4, default_sim_dt=1.0 / 120.0,
                )


class TestResolveViewportResolution:
    def test_none_keeps_default(self):
        assert resolve_viewport_resolution(
            None, default=(1280, 720),
        ) == (1280, 720)

    def test_parses_wxh(self):
        assert resolve_viewport_resolution(
            "960x540", default=(1280, 720),
        ) == (960, 540)

    def test_capital_x_and_whitespace(self):
        assert resolve_viewport_resolution(
            " 800X600 ", default=(1280, 720),
        ) == (800, 600)

    def test_missing_separator_raises(self):
        with pytest.raises(ValueError, match="WxH"):
            resolve_viewport_resolution("960", default=(1280, 720))

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="WxH"):
            resolve_viewport_resolution("axb", default=(1280, 720))

    def test_non_positive_raises(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_viewport_resolution("0x540", default=(1280, 720))
