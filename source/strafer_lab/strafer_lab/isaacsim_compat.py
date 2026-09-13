"""Compatibility layer for the ``isaacsim.core.utils`` surface deprecated at Isaac Sim 6.0.

Isaac Sim 6.0.0 GA deprecated ``isaacsim.core.api``, ``isaacsim.core.prims`` and
``isaacsim.core.utils`` in favour of ``isaacsim.core.experimental.*``, and the 6.0.x
line relocated them from ``isaacsim/exts/`` to ``isaacsim/extsDeprecated/``. Isaac Sim's
own Kit apps put that second directory on the extension search path; Isaac Lab's apps at
3.0.0-beta2 no longer do. Under ``isaaclab.sh -p`` the deprecated extensions are
therefore physically present but unimportable, and a direct
``from isaacsim.core.utils... import ...`` raises ``ModuleNotFoundError`` at runtime
rather than at collection.

This module routes the three symbols the repo uses through their documented
replacements. Each replacement ships under ``exts/`` on **both** the pre- and post-bump
pins, so every function here works on either, and the deprecated location is kept as a
fallback for any pin where a replacement is missing.

Call sites keep the deprecated call signatures. The adaptation lives here, once, rather
than as per-site ``try``/``except`` — which is what let one site
(``validate_scene_connectivity``) degrade silently for a whole release.

Imports are deferred into the function bodies on purpose. The replacements are Kit
extensions that resolve only once :class:`isaaclab.app.AppLauncher` has started
Omniverse, and this module must stay importable from a plain Python environment so the
pure test suite can assert against it without a Kit boot.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "add_labels",
    "enable_extension",
    "read_back_camera_anchor",
    "set_camera_view",
]

# Path used when a caller does not name a camera, matching the deprecated helper.
_DEFAULT_CAMERA_PRIM = "/OmniverseKit_Persp"


def enable_extension(name: str) -> None:
    """Enable a Kit extension by name, raising if it does not come up.

    Replacement: ``isaacsim.core.experimental.utils.app.enable_extension``.

    Both implementations return ``False`` rather than raising when Kit declines to enable
    the extension, and every call site in this repository discards that return. An
    extension that failed to enable produces *wrong* results downstream rather than
    missing ones — occupancy generation reads the omap extension's output, the bridge
    graph needs the ROS 2 bridge's node types registered — so the boolean is checked
    here, once, and turned into an exception. That closes every call site at the same
    time and is why this function returns nothing.

    The deprecated helper has no way to *disable* an extension, so no ``enabled``
    parameter is offered: a caller asking to disable would silently enable on the
    fallback path. Disabling has no user in this repository.

    Args:
        name: Extension name, e.g. ``"isaacsim.asset.gen.omap"``.

    Raises:
        RuntimeError: If the extension does not end up enabled, or if neither the
            replacement nor the deprecated module can be imported — which means Kit is
            not running, or is running without either reachable.
    """
    impl = None
    try:
        from isaacsim.core.experimental.utils.app import enable_extension as impl
    except ImportError:
        try:
            from isaacsim.core.utils.extensions import enable_extension as impl
        except ImportError as exc:
            raise RuntimeError(
                f"cannot enable Kit extension {name!r}: neither "
                "isaacsim.core.experimental.utils.app nor the deprecated "
                "isaacsim.core.utils.extensions is importable. This needs a running Kit "
                "runtime (launch through isaaclab.sh -p)."
            ) from exc
    if not impl(name):
        raise RuntimeError(
            f"Kit did not enable the extension {name!r}. Continuing would run against a "
            "missing extension, which yields wrong output rather than absent output."
        )


def _load_viewport_manager() -> Any | None:
    """Return ``ViewportManager``, enabling its extension first if need be.

    ``isaacsim.core.rendering_manager`` ships under ``exts/`` on both pins, but shipping
    is not the same as being loaded. Of Isaac Lab's six kit apps only
    ``isaaclab.python.headless.kit`` — the one :class:`~isaaclab.app.AppLauncher` selects
    for headless without cameras — leaves it out; the other five depend on it directly or
    inherit an app that does. Under that one app its Python directory is never placed on
    ``sys.path`` and the import fails even though the files are right there, so enabling
    it on demand is what makes the documented replacement usable everywhere. Note the
    shape of the trap: a check that the extension is present on disk passes while the
    call still fails at runtime.

    Returns ``None`` when the extension cannot be made available, so the caller can
    fall back rather than crash.
    """
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        return ViewportManager
    except ImportError:
        pass
    try:
        enable_extension("isaacsim.core.rendering_manager")
        from isaacsim.core.rendering_manager import ViewportManager

        return ViewportManager
    except (ImportError, RuntimeError):
        return None


# Larger than the replacement's own 1e-5 collinearity epsilon, so our nudge decides the
# roll instead of its own. At a 12 m overhead height this tilts the view by ~8e-6 rad.
_UP_AXIS_NUDGE_M = 1e-4


def _deconflict_up_axis(eye: Any, target: Any) -> Any:
    """Offset ``target`` along +Y when it sits directly below or above ``eye``.

    A look-at is undefined when the view direction is parallel to the up axis, and the
    two implementations resolve it differently. The deprecated helper wrote an explicit
    quaternion, leaving the camera's up vector at **+Y**. The replacement offsets the
    target by 1e-5 along **+X** and then does an ordinary look-at, which lands the up
    vector on **+X** — a 90 degree roll about the view axis.

    That is not cosmetic: this is exactly the pose the overhead followers in
    ``coverage_capture`` and ``teleop_capture`` request every step, so the difference
    would rotate recorded overhead video by a quarter turn.

    Offsetting along +Y instead reproduces the deprecated up vector through the
    replacement's own look-at, so the roll is preserved without reimplementing anyone's
    transform authoring. Non-degenerate poses are passed through untouched, and the
    equivalence is measured rather than assumed — see the compatibility smoke.
    """
    try:
        ex, ey = float(eye[0]), float(eye[1])
        tx, ty = float(target[0]), float(target[1])
    except (TypeError, IndexError, ValueError):
        return target
    if abs(tx - ex) > _UP_AXIS_NUDGE_M or abs(ty - ey) > _UP_AXIS_NUDGE_M:
        return target
    nudged = list(target)
    nudged[1] = ty + _UP_AXIS_NUDGE_M
    return nudged


def set_camera_view(
    eye: Any,
    target: Any,
    camera_prim_path: str = _DEFAULT_CAMERA_PRIM,
) -> None:
    """Point a camera prim at a target, going through Kit's transform command.

    Replacement: ``isaacsim.core.rendering_manager.ViewportManager.set_camera_view``,
    a classmethod taking the camera first and eye/target as keywords, where the
    deprecated free function took ``(eye, target, camera_prim_path)``. The argument
    order is reversed between them, which is the whole reason this wrapper keeps the
    old signature.

    ``viewport_api`` is deliberately not exposed: the replacement has no analogue, and
    no call site in this repo ever passed it.

    Two behavioural differences from the deprecated helper, both measured rather than
    inferred (see the compatibility smoke in this record's evidence deposit):

    * The deprecated helper **authors** ``omni:kit:centerOfInterest`` on the camera prim
      when it is absent, seeding it to ``(0, 0, -10)``. The replacement only reads that
      attribute, and leaves it absent if it was absent.
    * For a straight-down or straight-up view — ``eye`` and ``target`` sharing XY, which
      is exactly the overhead follow in ``coverage_capture`` and ``teleop_capture`` — the
      two disagree by a **90 degree roll** about the view axis: the deprecated helper
      writes an explicit quaternion leaving up at +Y, the replacement breaks the
      collinearity along +X and lands up on +X. :func:`_deconflict_up_axis` removes that
      difference; measured against the deprecated helper on the pin where both are
      reachable, the residual is 8e-06 on the up and forward vectors and 4e-06 on the
      quaternion, against exactly 0.0 for non-degenerate poses.

    Only the first of those is left standing, and no code in this repository reads
    ``omni:kit:centerOfInterest``.

    Args:
        eye: Camera position, as a sequence of three floats.
        target: Point the camera looks at, as a sequence of three floats.
        camera_prim_path: Prim path of the camera being posed.

    Raises:
        RuntimeError: If neither implementation is importable.
    """
    view_manager = _load_viewport_manager()
    if view_manager is not None:
        view_manager.set_camera_view(
            camera_prim_path, eye=eye, target=_deconflict_up_axis(eye, target)
        )
        return
    try:
        from isaacsim.core.utils.viewports import set_camera_view as _legacy
    except ImportError as exc:
        raise RuntimeError(
            "cannot set the camera view: neither isaacsim.core.rendering_manager nor "
            "the deprecated isaacsim.core.utils.viewports is importable. This needs a "
            "running Kit runtime (launch through isaaclab.sh -p)."
        ) from exc
    _legacy(eye=eye, target=target, camera_prim_path=camera_prim_path)


def add_labels(
    prim: Any,
    labels: list[str],
    instance_name: str = "class",
    overwrite: bool = True,
) -> None:
    """Apply UsdSemantics labels to a prim.

    Replacement: ``isaacsim.core.experimental.utils.semantics.add_labels``. The two
    differ in behaviour, not just in signature, and the difference is silent:

    * the deprecated helper takes ``overwrite`` and defaults to **replacing** the
      labels recorded for the taxonomy;
    * the replacement has no ``overwrite`` parameter and always **appends**,
      de-duplicating against whatever is already there.

    So a re-run over an already-labelled stage accumulates under the replacement where
    it used to replace. This wrapper preserves the deprecated contract by clearing the
    taxonomy's existing labels first when ``overwrite`` is set, leaving other taxonomies
    on the prim untouched.

    Args:
        prim: Prim path or ``Usd.Prim`` to label.
        labels: Labels to apply.
        instance_name: Taxonomy (semantic instance) the labels belong to.
        overwrite: Replace the taxonomy's existing labels rather than adding to them.

    Raises:
        RuntimeError: If neither implementation is importable.
    """
    try:
        from isaacsim.core.experimental.utils import semantics as _semantics
    except ImportError:
        _semantics = None

    if _semantics is not None:
        if overwrite:
            existing = _semantics.get_labels(prim).get(instance_name, [])
            if existing:
                _semantics.remove_labels(prim, labels=existing, taxonomy=instance_name)
        _semantics.add_labels(prim, labels=labels, taxonomy=instance_name)
        return

    try:
        from isaacsim.core.utils.semantics import add_labels as _legacy
    except ImportError as exc:
        raise RuntimeError(
            "cannot apply semantic labels: neither "
            "isaacsim.core.experimental.utils.semantics nor the deprecated "
            "isaacsim.core.utils.semantics is importable. Applying labels needs the "
            "Isaac Sim Kit runtime (launch through isaaclab.sh -p)."
        ) from exc
    _legacy(prim, labels, instance_name=instance_name, overwrite=overwrite)


def read_back_camera_anchor(
    env: Any,
    requested_eye: Any,
    requested_target: Any,
    tolerance: float = 1e-3,
    strict: bool = True,
) -> dict[str, Any]:
    """Read back where the video recorder's camera actually ended up.

    Writing ``eye``/``lookat`` (or the pre-3.0.0-beta2 ``camera_position``/
    ``camera_target``) on the capture config proves only that the field exists. The
    recorder applies the config itself, on the first render, inside
    ``IsaacsimKitPerspectiveVideo.render_rgb_array``: the branch that builds the RGB
    annotator also poses the camera prim from ``cfg``. A pose read before that call is
    not the pose the clip was filmed from, and a write that landed on a field nothing
    reads leaves no trace in the logs at all — which is how two pins were compared from
    different framings and read as a photometric shift.

    So this forces exactly one render, then reads the camera prim's world transform off
    the stage and checks it against what was asked for. Two things are checked, because
    position alone does not pin a camera: the eye must land within ``tolerance``, and the
    view ray from the observed eye along the observed forward must pass through the
    requested target within ``tolerance``.

    Args:
        env: The unwrapped environment (the one owning ``video_recorder`` and ``sim``).
        requested_eye: World-space camera position the caller anchored on.
        requested_target: World-space point the caller aimed at.
        tolerance: Metres of slack on both the eye and the target-ray checks.
        strict: Raise when the observed pose disagrees. Pass ``False`` for a camera the
            caller re-poses itself every step — the overhead follow in
            ``coverage_capture`` is the case: there the setup pose is meant to be
            overridden, so a disagreement is information, not a fault.

    Returns:
        The anchor record: requested and observed poses, the field pair actually
        written, the camera prim path, whether the pose matched, and the Isaac Lab
        version. Callers write this beside the clip so a later comparison can prove the
        two clips share a framing.

    Raises:
        RuntimeError: If no capture object is reachable, if the stage has no camera prim
            at the configured path, or — when ``strict`` — if the observed pose
            disagrees with the request.
    """
    from pxr import Gf, Usd, UsdGeom

    recorder = getattr(env, "video_recorder", None)
    capture = getattr(recorder, "_capture", None) if recorder is not None else None
    if capture is None:
        raise RuntimeError(
            "--video: no capture object on the environment's video recorder, so the "
            "recording cannot be anchored and the clip would be filmed from the "
            "recorder's own default pose. `_capture` is a private Isaac Lab attribute; "
            "if it has been renamed upstream, this call site needs updating."
        )

    prim_path = getattr(capture.cfg, "camera_prim_path", _DEFAULT_CAMERA_PRIM)
    if hasattr(capture.cfg, "eye"):
        field_pair = "eye/lookat"
    elif hasattr(capture.cfg, "camera_position"):
        field_pair = "camera_position/camera_target"
    else:
        field_pair = "none"

    # The pose is applied by the recorder, not by the write above, and only on its
    # first render. Anything read before this call describes the wrong camera.
    env.render()

    stage = env.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(
            f"--video: no camera prim at {prim_path!r} after the first render, so the "
            "recorder's pose cannot be read back."
        )

    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    observed_eye = xform.ExtractTranslation()
    # USD cameras look down their local -Z.
    forward = xform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()

    want_eye = Gf.Vec3d(*(float(v) for v in requested_eye))
    want_target = Gf.Vec3d(*(float(v) for v in requested_target))
    eye_error = (Gf.Vec3d(*observed_eye) - want_eye).GetLength()

    # Distance from the requested target to the observed view ray.
    to_target = want_target - Gf.Vec3d(*observed_eye)
    along = Gf.Dot(to_target, forward)
    target_error = (to_target - forward * along).GetLength()

    record = {
        "camera_prim_path": prim_path,
        "field_pair_written": field_pair,
        "requested_eye": [float(v) for v in requested_eye],
        "requested_target": [float(v) for v in requested_target],
        "observed_eye": [float(v) for v in observed_eye],
        "observed_forward": [float(v) for v in forward],
        "eye_error_m": float(eye_error),
        "target_ray_error_m": float(target_error),
        "tolerance_m": float(tolerance),
        "matched": bool(eye_error <= tolerance and target_error <= tolerance),
        "isaaclab_version": _isaaclab_version(),
    }

    if not record["matched"] and strict:
        raise RuntimeError(
            "--video: the recorder filmed from a different pose than the one anchored "
            f"on env 0. Requested eye {record['requested_eye']} target "
            f"{record['requested_target']}; observed eye {record['observed_eye']} "
            f"forward {record['observed_forward']} (eye off by {eye_error:.4f} m, "
            f"target ray off by {target_error:.4f} m). The write went to "
            f"{field_pair!r} on {type(capture.cfg).__name__}. A clip filmed this way "
            "cannot be compared photometrically against one filmed from the anchor."
        )
    return record


def _isaaclab_version() -> str:
    """Version of the installed Isaac Lab, or ``unknown`` if it cannot be determined."""
    try:
        import importlib.metadata as _md

        return _md.version("isaaclab")
    except Exception:
        return "unknown"
