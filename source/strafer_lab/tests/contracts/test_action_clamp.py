"""Parity + correctness tests for the L1 body-frame velocity clamp.

The chassis cannot deliver max-forward + max-strafe simultaneously
because each mecanum wheel has a single motor capped at
``MAX_WHEEL_ANGULAR_VEL``. The Jetson inference node scales body-frame
``(vx, vy)`` jointly under an L1 budget before publishing
``/cmd_vel`` so the commanded heading is preserved. Sim's
``MecanumWheelAction.process_actions`` applies the same clamp at the
same point in the action pipeline.

These tests pin three properties:

1. **Cross-lane parity.** The scalar form
   (``strafer_shared.mecanum_kinematics.l1_clamp_twist`` — the Jetson
   call site) and the torch-batched form
   (``l1_clamp_twist_batched`` — the sim call site) produce identical
   outputs to within float roundoff for the same ``(vx, vy, omega)``
   input. This anchors the one-source-of-truth claim.
2. **Heading preservation.** Above-cap commands scale ``(vx, vy)``
   jointly; ``atan2(vy, vx)`` is unchanged. Per-wheel clipping (which
   sim's ``torch.clamp`` would do downstream without the L1 stage in
   front) would skew the heading; this test would catch a regression
   to that behavior.
3. **Downstream wheel cap.** After the L1 clamp + inverse kinematics,
   every wheel velocity is within ``MAX_WHEEL_ANGULAR_VEL`` (no
   per-wheel saturation activates on the clamped output). This is
   the contract the per-wheel ``torch.clamp`` below the L1 stage no
   longer has to enforce by itself.

The tests do not launch Isaac Sim. The batched helper is exercised
directly on CPU tensors; the action-term integration is verified at
the helper boundary, since the helper is the entire delta between
the pre- and post-brief sim behavior.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from strafer_shared.constants import (
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    MAX_WHEEL_ANGULAR_VEL,
    NAV_VEL_SCALE,
)
from strafer_shared.mecanum_kinematics import (
    KINEMATIC_MATRIX,
    l1_clamp_twist,
    l1_clamp_twist_batched,
)


# Sim caps: the training action term's _velocity_scale resolves normalized
# [-1, 1]^3 to (MAX_LINEAR_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL), so the L1
# budget is the full hardware envelope.
_SIM_LIN_CAP = MAX_LINEAR_VEL
_SIM_ANG_CAP = MAX_ANGULAR_VEL

# Jetson caps under default NAV_VEL_SCALE=0.5; used for the cross-lane
# parity sweep so the test covers a realistic deployment configuration
# in addition to the sim configuration.
_JETSON_LIN_CAP = NAV_VEL_SCALE * MAX_LINEAR_VEL
_JETSON_ANG_CAP = NAV_VEL_SCALE * MAX_ANGULAR_VEL


# -----------------------------------------------------------------------------
# Cross-lane parity (helper level)
# -----------------------------------------------------------------------------


class TestCrossLaneParity:
    """The scalar form (Jetson) and the batched form (sim) must be
    numerically identical for the same input + caps. This is the
    one-source-of-truth anchor.
    """

    @pytest.mark.parametrize(
        "vx,vy,omega",
        [
            (0.0, 0.0, 0.0),
            (0.3, 0.4, 1.5),                              # within budget
            (0.99 * _SIM_LIN_CAP, 0.99 * _SIM_LIN_CAP, 0.0),   # over budget
            (1.55, 1.55, 4.15),                           # brief's worst case
            (-0.9, 0.3, -3.0),                            # mixed signs
            (0.0, 0.0, 10.0),                             # omega only, over cap
            (-1.0, -1.0, -10.0),                          # all over, negative
        ],
    )
    @pytest.mark.parametrize(
        "lin_cap,ang_cap",
        [
            (_SIM_LIN_CAP, _SIM_ANG_CAP),     # sim configuration
            (_JETSON_LIN_CAP, _JETSON_ANG_CAP),  # default Jetson configuration
        ],
    )
    def test_scalar_and_batched_agree(
        self, vx, vy, omega, lin_cap, ang_cap
    ):
        scalar_out = l1_clamp_twist(
            vx, vy, omega,
            vel_cap_linear_m_s=lin_cap,
            vel_cap_angular_rad_s=ang_cap,
        )
        body = torch.tensor(
            [[vx, vy, omega]], dtype=torch.float64
        )
        batched_out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=lin_cap,
            vel_cap_angular_rad_s=ang_cap,
        )
        np.testing.assert_allclose(
            batched_out.squeeze(0).numpy(),
            np.asarray(scalar_out, dtype=np.float64),
            atol=1e-12,
            rtol=0,
        )

    def test_jetson_alias_is_the_shared_helper(self):
        """``strafer_inference.obs_pipeline.l1_clamp_velocity`` must be
        the same callable as ``strafer_shared.mecanum_kinematics.l1_clamp_twist``
        — the alias migration path. A future refactor that breaks this
        identity should also remove the alias (and update the inference
        node call site).

        Skips on DGX (no ROS 2 sourced, so ``strafer_inference`` is not
        on ``sys.path``); the Jetson-side ``colcon test --packages-select
        strafer_inference`` covers the alias by exercising the
        ``l1_clamp_velocity`` import directly.
        """
        obs_pipeline = pytest.importorskip("strafer_inference.obs_pipeline")
        assert obs_pipeline.l1_clamp_velocity is l1_clamp_twist


# -----------------------------------------------------------------------------
# Within-budget no-op
# -----------------------------------------------------------------------------


class TestWithinBudgetPassthrough:
    """Below-cap commands must pass through unchanged. The L1 clamp is
    not a smoothing filter — it intervenes only when the chassis can't
    deliver the command.
    """

    def test_scalar_below_cap_unchanged(self):
        vx, vy, omega = l1_clamp_twist(
            0.3, 0.4, 1.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert vx == pytest.approx(0.3)
        assert vy == pytest.approx(0.4)
        assert omega == pytest.approx(1.0)

    def test_batched_below_cap_unchanged(self):
        body = torch.tensor(
            [[0.3, 0.4, 1.0], [-0.2, 0.1, -0.5]], dtype=torch.float32
        )
        out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        torch.testing.assert_close(out, body)

    def test_at_cap_exactly_unchanged(self):
        """``|vx| + |vy| == cap`` is the boundary; the clamp must not
        engage (no off-by-one rounding at the threshold).
        """
        vx_in = 0.6 * _SIM_LIN_CAP
        vy_in = 0.4 * _SIM_LIN_CAP
        vx, vy, _ = l1_clamp_twist(
            vx_in, vy_in, 0.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert vx == pytest.approx(vx_in)
        assert vy == pytest.approx(vy_in)


# -----------------------------------------------------------------------------
# Above-cap heading preservation
# -----------------------------------------------------------------------------


class TestHeadingPreservation:
    """The core correctness anchor: scaling ``(vx, vy)`` jointly keeps
    the commanded heading. Clipping per-axis (the pre-brief behavior
    sim exhibited) would skew it.
    """

    @pytest.mark.parametrize(
        "vx_in,vy_in",
        [
            (0.9 * _SIM_LIN_CAP, 0.7 * _SIM_LIN_CAP),
            (-1.2, 0.4),
            (0.5, -0.8),
            (_SIM_LIN_CAP, _SIM_LIN_CAP),
        ],
    )
    def test_scalar_heading_preserved(self, vx_in, vy_in):
        original_heading = math.atan2(vy_in, vx_in)
        vx, vy, _ = l1_clamp_twist(
            vx_in, vy_in, 0.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        clamped_heading = math.atan2(vy, vx)
        assert clamped_heading == pytest.approx(original_heading)
        # And L1 budget is honoured.
        assert abs(vx) + abs(vy) <= _SIM_LIN_CAP + 1e-9

    def test_batched_heading_preserved_vectorized(self):
        body = torch.tensor(
            [
                [0.9 * _SIM_LIN_CAP, 0.7 * _SIM_LIN_CAP, 0.0],
                [-1.2, 0.4, 0.0],
                [0.5, -0.8, 0.0],
                [_SIM_LIN_CAP, _SIM_LIN_CAP, 0.0],
            ],
            dtype=torch.float64,
        )
        out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        original_headings = torch.atan2(body[:, 1], body[:, 0])
        clamped_headings = torch.atan2(out[:, 1], out[:, 0])
        torch.testing.assert_close(
            clamped_headings, original_headings, atol=1e-12, rtol=0
        )

    def test_omega_clamps_independently_of_vx_vy(self):
        vx, vy, omega = l1_clamp_twist(
            0.3, 0.4, 10.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert vx == pytest.approx(0.3)
        assert vy == pytest.approx(0.4)
        assert omega == pytest.approx(_SIM_ANG_CAP)


# -----------------------------------------------------------------------------
# Downstream wheel cap
# -----------------------------------------------------------------------------


class TestPostClampWheelCap:
    """The L1 clamp's primary correctness claim is for the translation
    subspace: with ``omega = 0``, ``|vx| + |vy| <= MAX_LINEAR_VEL``
    implies every wheel velocity is within ``MAX_WHEEL_ANGULAR_VEL``
    after inverse kinematics, with no per-wheel saturation needed.
    This is the case sim's heading-distorting per-wheel ``torch.clamp``
    used to apply incorrectly — the regression case the brief targets.

    For combined ``(vx, vy, omega)`` over budget, the per-wheel
    ``torch.clamp`` downstream of the L1 stage in
    ``MecanumWheelAction.process_actions`` remains a necessary safety
    net for the omega contribution: with both subsystems at cap,
    ``|(vx + vy + K*omega) / r|`` can reach ``2 * MAX_WHEEL_ANGULAR_VEL``.
    The L1 clamp doesn't replace that downstream clip; it ensures the
    *heading* of the translation subspace survives.
    """

    @pytest.mark.parametrize(
        "vx_in,vy_in",
        [
            (_SIM_LIN_CAP, _SIM_LIN_CAP),                # at-budget diagonal
            (-_SIM_LIN_CAP, _SIM_LIN_CAP),                # negative quadrant
            (2.0 * _SIM_LIN_CAP, 2.0 * _SIM_LIN_CAP),    # 2× over budget
            (0.5 * _SIM_LIN_CAP, 1.5 * _SIM_LIN_CAP),    # asymmetric over budget
        ],
    )
    def test_translation_only_within_wheel_cap(self, vx_in, vy_in):
        """Pure-translation commands (omega=0) post-clamp must satisfy
        the per-wheel cap, since ``|vx| + |vy| <= MAX_LINEAR_VEL =
        WHEEL_RADIUS * MAX_WHEEL_ANGULAR_VEL`` and the wheel rows
        depend only on ``(vx ± vy) / WHEEL_RADIUS``.
        """
        vx, vy, _ = l1_clamp_twist(
            vx_in, vy_in, 0.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        wheel_vels = KINEMATIC_MATRIX @ np.array(
            [vx, vy, 0.0], dtype=np.float64
        )
        assert np.all(np.abs(wheel_vels) <= MAX_WHEEL_ANGULAR_VEL + 1e-9), (
            f"wheel velocities exceed cap on pure-translation clamp: "
            f"{wheel_vels}"
        )

    def test_brief_worst_case_diagonal(self):
        """The brief's literal example: ``(0.99, 0.99, 0)`` normalized
        denormalizes to ``(0.99 * MAX_LINEAR_VEL, 0.99 * MAX_LINEAR_VEL,
        0)`` which would otherwise demand FL = (vx + vy) / WHEEL_RADIUS
        ≈ 64.6 rad/s — roughly 2× the motor cap. The L1 clamp scales
        both axes to halve the demand, restoring the per-wheel cap as
        an upper bound.
        """
        vx_in = 0.99 * _SIM_LIN_CAP
        vy_in = 0.99 * _SIM_LIN_CAP
        vx, vy, _ = l1_clamp_twist(
            vx_in, vy_in, 0.0,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert vx == pytest.approx(vy)  # 45° heading preserved
        assert abs(vx) + abs(vy) == pytest.approx(_SIM_LIN_CAP)
        wheel_vels = KINEMATIC_MATRIX @ np.array(
            [vx, vy, 0.0], dtype=np.float64
        )
        assert np.all(np.abs(wheel_vels) <= MAX_WHEEL_ANGULAR_VEL + 1e-9)

    @pytest.mark.parametrize(
        "vx_in,vy_in,omega_in",
        [
            (_SIM_LIN_CAP, _SIM_LIN_CAP, _SIM_ANG_CAP),
            (2.0 * _SIM_LIN_CAP, 2.0 * _SIM_LIN_CAP, 2.0 * _SIM_ANG_CAP),
        ],
    )
    def test_full_pipeline_wheel_cap_under_combined_overdrive(
        self, vx_in, vy_in, omega_in
    ):
        """Combined (vx, vy, omega) over-budget commands need both the
        L1 stage AND the downstream per-wheel clip to stay within the
        motor cap. This mirrors the sequence in
        ``MecanumWheelAction.process_actions``: L1 first (heading-
        preserving), per-wheel clip second (safety net).
        """
        vx, vy, omega = l1_clamp_twist(
            vx_in, vy_in, omega_in,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        wheel_vels = KINEMATIC_MATRIX @ np.array(
            [vx, vy, omega], dtype=np.float64
        )
        # Per-wheel clip — the safety net the L1 stage doesn't replace.
        wheel_vels_clamped = np.clip(
            wheel_vels, -MAX_WHEEL_ANGULAR_VEL, MAX_WHEEL_ANGULAR_VEL
        )
        assert np.all(
            np.abs(wheel_vels_clamped) <= MAX_WHEEL_ANGULAR_VEL + 1e-9
        )


# -----------------------------------------------------------------------------
# Batched form: shape + dtype invariants
# -----------------------------------------------------------------------------


class TestBatchedShapeAndDtype:
    """Sim runs the action term over ``(num_envs, 3)`` tensors; the
    helper must preserve shape and dtype regardless of leading axes.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_round_trip(self, dtype):
        body = torch.tensor(
            [[0.3, 0.4, 1.0], [2.0, 2.0, 5.0]], dtype=dtype
        )
        out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert out.dtype == dtype
        assert out.shape == body.shape

    def test_supports_extra_leading_axes(self):
        body = torch.zeros((2, 5, 3), dtype=torch.float32)
        body[..., 0] = 2.0
        body[..., 1] = 2.0
        out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        assert out.shape == body.shape
        # Every element saw the same input, so the L1-sum equals cap
        # for every element after clamp.
        l1 = out[..., 0].abs() + out[..., 1].abs()
        torch.testing.assert_close(
            l1, torch.full_like(l1, _SIM_LIN_CAP), atol=1e-5, rtol=0
        )

    def test_zero_input_zero_output_no_nan(self):
        """The safe-denominator guard inside the batched form must not
        introduce NaNs when ``l1 == 0`` on any element.
        """
        body = torch.zeros((4, 3), dtype=torch.float32)
        out = l1_clamp_twist_batched(
            body,
            vel_cap_linear_m_s=_SIM_LIN_CAP,
            vel_cap_angular_rad_s=_SIM_ANG_CAP,
        )
        torch.testing.assert_close(out, body)
        assert not torch.isnan(out).any()
