"""Pure resolvers for the teleop perf knobs (decimation + viewport size).

Profiling the teleop loop on a high-density Infinigen scene showed the
per-tick wall-time is dominated by PhysX (the ``decimation`` physics
substeps) and the Kit viewport render, with the env step's manager loop a
distant third. These two knobs attack those two terms; the resolvers here
are split out as pure functions so they unit-test without the Isaac Sim
runtime.

Neither knob touches the captured dataset: ``decimation`` is rescaled to
hold the env step (control / action) rate constant, and the viewport is
the operator-only editor view that never enters a render product.
"""

from __future__ import annotations


def resolve_decimation(
    requested: int | None,
    *,
    default_decimation: int,
    default_sim_dt: float,
) -> tuple[int, float]:
    """Resolve a requested physics ``decimation`` and the matching ``sim.dt``.

    ``env.step`` runs ``decimation`` PhysX substeps of ``sim.dt`` each, so
    one env step spans ``decimation * sim_dt`` of sim time and the env step
    (control) rate is ``1 / (decimation * sim_dt)``. Fewer substeps is the
    only lever that cuts PhysX cost, but naively lowering ``decimation``
    alone would also speed up the control rate and change the action
    contract. So we **hold ``decimation * sim_dt`` constant**: the env step
    rate, the captured action cadence, and sim-time-per-step are unchanged;
    only the physics substep count (and thus PhysX wall-cost) drops, at the
    cost of coarser contact resolution per substep.

    Returns ``(decimation, sim_dt)``. ``requested=None`` keeps the env's
    defaults unchanged. Raises ``ValueError`` for a non-positive request.
    """
    if requested is None:
        return default_decimation, default_sim_dt
    if requested <= 0:
        raise ValueError(f"decimation must be positive, got {requested}")
    # Hold the control rate: new_dt * new_decim == default_dt * default_decim.
    control_step_s = default_sim_dt * default_decimation
    new_sim_dt = control_step_s / requested
    return requested, new_sim_dt


def resolve_viewport_resolution(
    spec: str | None,
    *,
    default: tuple[int, int],
) -> tuple[int, int]:
    """Parse a ``WxH`` viewport-resolution spec to a ``(width, height)`` tuple.

    The viewport is the operator-only editor view; lowering it cuts the Kit
    render cost without touching any captured frame (the perception camera
    render product is a separate surface at its own fixed resolution).

    Accepts ``"960x540"`` / ``"960X540"`` (case-insensitive ``x``).
    ``spec=None`` returns ``default``. Raises ``ValueError`` on a malformed
    or non-positive spec.
    """
    if spec is None:
        return default
    text = spec.strip().lower().replace(" ", "")
    if "x" not in text:
        raise ValueError(
            f"viewport resolution must be 'WxH' (e.g. 960x540), got {spec!r}",
        )
    w_str, _, h_str = text.partition("x")
    try:
        width, height = int(w_str), int(h_str)
    except ValueError:
        raise ValueError(
            f"viewport resolution must be 'WxH' integers, got {spec!r}",
        ) from None
    if width <= 0 or height <= 0:
        raise ValueError(
            f"viewport resolution must be positive, got {width}x{height}",
        )
    return width, height
