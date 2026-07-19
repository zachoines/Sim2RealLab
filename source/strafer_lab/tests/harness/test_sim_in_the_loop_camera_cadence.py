"""The bridge derives its camera publish cadence to one frame per policy period.

These tests pin the cadence contract: the bridge publishes one camera frame per
policy period (1/30 sim-s = 30 Hz), the rate the real D555 and training (a fresh
render per env.step) both satisfy. The frame skip that lands that cadence is
derived Kit-free from the effective ``sim.dt`` x ``decimation`` (3 at decimation
1, 0 at decimation 4) and floors at 0; the ``--camera-frame-skip`` flag is an
explicit escape hatch that wins verbatim (floored at 0); and two startup
warnings flag an off-policy explicit skip and a render_interval too coarse to
feed the cadence (duplicate frames). ``default=None`` on the flag is the sentinel
that selects derivation, so it is asserted directly against the parser.

Hermetic: no Kit boot, no live env. ``run_sim_in_the_loop`` is imported inside
each test (it pulls ``isaaclab.app`` at module top), the same in-script pattern
``test_sim_in_the_loop_terminations.py`` uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SIM_DT = 1.0 / 120.0  # the trained-policy physics dt (strafer_shared POLICY_SIM_DT)


def _import_rsil():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_sim_in_the_loop as rsil

    return rsil


class TestDeriveCameraFrameSkip:
    def test_derives_correct_skip_across_decimation_matrix(self):
        rsil = _import_rsil()
        # decimation 1: 120 Hz bridge ticks throttled to the 30 Hz contract.
        assert rsil._derive_camera_frame_skip(SIM_DT, 1) == 3
        # decimation 4: a bridge tick is already the 30 Hz policy period.
        assert rsil._derive_camera_frame_skip(SIM_DT, 4) == 0
        # decimation 2: 60 Hz bridge ticks -> skip every other one.
        assert rsil._derive_camera_frame_skip(SIM_DT, 2) == 1

    def test_floors_at_zero_when_tick_slower_than_policy_period(self):
        rsil = _import_rsil()
        # decimation 8 -> 15 Hz bridge ticks, already slower than 30 Hz;
        # skipping cannot speed it up, so publish every tick.
        assert rsil._derive_camera_frame_skip(SIM_DT, 8) == 0
        assert rsil._derive_camera_frame_skip(SIM_DT, 6) == 0

    def test_derivation_follows_sim_dt_not_a_hardcoded_120hz(self):
        rsil = _import_rsil()
        # At a 60 Hz physics dt the same policy period needs a different skip.
        assert rsil._derive_camera_frame_skip(1.0 / 60.0, 1) == 1
        assert rsil._derive_camera_frame_skip(1.0 / 60.0, 2) == 0

    def test_policy_period_override_respected(self):
        rsil = _import_rsil()
        # A 15 Hz policy period at 120 Hz / decimation 1: 8 ticks per period.
        assert (
            rsil._derive_camera_frame_skip(SIM_DT, 1, policy_period_s=1.0 / 15.0) == 7
        )


class TestResolveCameraFrameSkip:
    def test_unset_derives_and_flags_not_explicit(self):
        rsil = _import_rsil()
        skip, explicit = rsil._resolve_camera_frame_skip(None, sim_dt=SIM_DT, decimation=4)
        assert skip == 0
        assert explicit is False

    def test_explicit_value_wins_verbatim_even_off_policy(self):
        rsil = _import_rsil()
        # Operator forces skip 3 at decimation 4 (e.g. a capacity-limited
        # consumer): the value is honored, not overridden to the derived 0.
        skip, explicit = rsil._resolve_camera_frame_skip(3, sim_dt=SIM_DT, decimation=4)
        assert skip == 3
        assert explicit is True

    def test_explicit_zero_is_explicit_not_derived(self):
        rsil = _import_rsil()
        # 0 is a real value, distinct from the unset (None) sentinel.
        skip, explicit = rsil._resolve_camera_frame_skip(0, sim_dt=SIM_DT, decimation=1)
        assert skip == 0
        assert explicit is True

    def test_explicit_negative_floored_to_zero(self):
        rsil = _import_rsil()
        # A stray negative must floor to 0 (matching the publisher), not reach
        # the cadence report as -1 and drive publish_period_ticks to 0.
        skip, explicit = rsil._resolve_camera_frame_skip(-1, sim_dt=SIM_DT, decimation=1)
        assert skip == 0
        assert explicit is True


class TestCameraCadence:
    def test_default_decimation1_reports_30hz_publish_no_warnings(self):
        rsil = _import_rsil()
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=1, render_interval=1, frame_skip=3, explicit=False
        )
        assert cad.publish_hz == pytest.approx(30.0)
        assert cad.bridge_tick_hz == pytest.approx(120.0)
        assert cad.renders_per_tick == pytest.approx(1.0)
        assert cad.derived_frame_skip == 3
        assert cad.warnings == ()

    def test_decimation4_derived_reports_30hz_publish_no_warnings(self):
        rsil = _import_rsil()
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=4, render_interval=4, frame_skip=0, explicit=False
        )
        assert cad.publish_hz == pytest.approx(30.0)
        assert cad.bridge_tick_hz == pytest.approx(30.0)
        assert cad.renders_per_tick == pytest.approx(1.0)
        assert cad.warnings == ()

    def test_explicit_off_policy_skip_warns_with_effective_rate(self):
        rsil = _import_rsil()
        # An off-policy explicit skip: 3 at decimation 4 -> 7.5 Hz depth, a
        # quarter of the 30 Hz policy period.
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=4, render_interval=4, frame_skip=3, explicit=True
        )
        assert cad.publish_hz == pytest.approx(7.5)
        assert cad.derived_frame_skip == 0
        assert len(cad.warnings) == 1
        assert "7.50 Hz" in cad.warnings[0]
        assert "30.00 Hz policy period" in cad.warnings[0]

    def test_explicit_matching_derived_does_not_warn(self):
        rsil = _import_rsil()
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=1, render_interval=1, frame_skip=3, explicit=True
        )
        assert cad.warnings == ()

    def test_derived_value_never_raises_off_policy_warning(self):
        rsil = _import_rsil()
        # A derived (explicit=False) config that DOES emit a warning (the
        # coarse-render duplicate-frame one) — so the assertion is not vacuous:
        # the off-policy discriminator must stay silent on the derived path even
        # while another warning fires.
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=4, render_interval=8, frame_skip=0, explicit=False
        )
        assert len(cad.warnings) == 1
        assert all("--camera-frame-skip" not in w for w in cad.warnings)

    def test_coarse_render_interval_warns_duplicate_frames(self):
        rsil = _import_rsil()
        # decimation 4, render every 8 physics ticks -> 0.5 renders per bridge
        # tick; frame_skip 0 (the derived value) publishes faster than the
        # render advances, so half the frames are stale duplicates.
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=4, render_interval=8, frame_skip=0, explicit=False
        )
        assert cad.renders_per_tick == pytest.approx(0.5)
        assert len(cad.warnings) == 1
        assert "duplicate frames" in cad.warnings[0]

    def test_default_config_does_not_warn_duplicate_frames(self):
        rsil = _import_rsil()
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=1, render_interval=1, frame_skip=3, explicit=False
        )
        assert all("duplicate frames" not in w for w in cad.warnings)

    def test_both_warnings_fire_independently(self):
        rsil = _import_rsil()
        # An explicit off-policy skip AND a render_interval too coarse to feed
        # it: both conditions must fire (the two branches are independent, not
        # mutually exclusive). decimation 4, render_interval 16, frame_skip 1
        # (derived is 0) -> off-policy warns; renders_per_publish 0.5 -> dup warns.
        cad = rsil._camera_cadence(
            sim_dt=SIM_DT, decimation=4, render_interval=16, frame_skip=1, explicit=True
        )
        assert len(cad.warnings) == 2
        joined = " ".join(cad.warnings)
        assert "--camera-frame-skip" in joined
        assert "duplicate frames" in joined


class TestParseArgsCameraFrameSkipDefault:
    def test_omitted_flag_parses_to_none_sentinel(self, monkeypatch):
        rsil = _import_rsil()
        # The derived-cadence fix hinges on the default being the None sentinel
        # (not 0, not 3): a silent revert to a fixed default would re-break
        # decimation-4 cadence while every helper test — which passes values by
        # hand — stayed green. Pin it against the real parser.
        monkeypatch.setattr(sys, "argv", ["run_sim_in_the_loop.py"])
        args = rsil._parse_args()
        assert args.camera_frame_skip is None

    def test_explicit_flag_parses_to_int(self, monkeypatch):
        rsil = _import_rsil()
        monkeypatch.setattr(sys, "argv", ["run_sim_in_the_loop.py", "--camera-frame-skip", "3"])
        args = rsil._parse_args()
        assert args.camera_frame_skip == 3
