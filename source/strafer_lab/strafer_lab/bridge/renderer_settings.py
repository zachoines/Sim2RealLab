"""Renderer selection for the sim-in-the-loop bridge (RTX Real-Time 2.0 A/B).

Pure Python — no Kit / ``omni.*`` / ``isaaclab`` import, so the launch-arg
resolution is unit-testable without booting Isaac Sim (same ethos as
:mod:`strafer_lab.bridge.config`).

The bridge defaults to the **RTX Real-Time 2.0** renderer
(``/rtx/rendermode = "RealTimePathTracing"``) in Isaac Sim's **Performance**
render-quality preset. ``--renderer legacy`` reverts to the Real-Time 1.0
(``RaytracedLighting``) renderer for visual debugging where deterministic
shading matters. The ``legacy`` override is a **temporary A/B toggle**: it
exists to de-risk the default flip and is expected to be removed once Real-Time
2.0 is validated on the rig — it is not a permanent renderer lane.

Two deliberate design points, both load-bearing:

1. **Set at app-launch, not post-boot.** Real-Time 2.0 must be the *active*
   render mode AND its renderer must be *registered* at startup. Registration
   is gated by ``/rtx-transient/rt2Enabled`` /
   ``/persistent/rtx/modes/rt2/enabled``, which the RTX renderer reads at boot
   (a persistent preference, not a live toggle). Pushing ``/rtx/rendermode``
   through the sim's ``RenderCfg.carb_settings`` *after* the app has booted is
   therefore not reliable for the mode switch — so these settings are injected
   as Kit CLI args before ``AppLauncher`` starts. (Post-boot ``carb_settings``
   remains the right home for dynamic render settings like RTX auto-exposure.)

2. **No DLSS frame generation ("FPS multiplier").** The renderer's DLSS-G frame
   generation is deliberately left OFF (``1x``, ``dlssg.enabled = false``):

   - It is a swapchain/**present-path** feature. The headless daily-driver
     bridge disables the present thread outright, so it is a no-op there; and
     the camera publisher reads the offscreen ``TiledCamera`` annotator output
     once per ``env.step`` (an env-step-driven cadence), never the presented
     stream — frame generation has no path to inject, duplicate, or fabricate a
     ``/d555/depth/image_rect_raw`` frame that feeds the depth policy.
   - Isaac Sim's ``Performance`` preset already disables it with the note that
     it "does not yet support tiled camera well", and the base app disables it
     as "not compatible with synthetic data generation" — the perception camera
     is exactly a tiled synthetic-data render product.
   - The x3/x4 multiplier is an opaque internal (hashed) Kit setting with no
     stable carb path and no ``RenderCfg`` field, so "4x" is not cleanly
     expressible in any case.

   We force it off explicitly so the decision is visible in the boot args, not
   merely inherited from a preset default.
"""

from __future__ import annotations

RENDERER_RT2 = "rt2"
RENDERER_LEGACY = "legacy"
RENDERER_CHOICES = (RENDERER_RT2, RENDERER_LEGACY)
DEFAULT_RENDERER = RENDERER_RT2

# RTX render-mode carb tokens. "RealTimePathTracing" is the Real-Time 2.0 mode
# (Isaac Sim's own SimulationApp default); "RaytracedLighting" is the legacy
# Real-Time 1.0 mode (deprecated in the Kit renderer menu).
RT2_RENDERMODE_TOKEN = "RealTimePathTracing"
LEGACY_RENDERMODE_TOKEN = "RaytracedLighting"

# Isaac Sim render-quality preset loaded for the RT 2.0 default. "performance"
# maps to apps/rendering_modes/performance.kit (RT2 path-tracer tuning, DLSS
# exec mode 0 = Performance, frame-gen off). Legacy leaves the app's default
# preset untouched so a debug session sees stock shading.
RT2_RENDERING_MODE = "performance"


def _rt2_kit_setting_overrides() -> "dict[str, object]":
    """Kit settings that force Real-Time 2.0 active + registered at startup.

    Ordered dict (insertion order is preserved in the emitted CLI args):

    - ``/rtx/rendermode`` selects the active render mode.
    - ``/rtx-transient/rt2Enabled`` + ``/persistent/rtx/modes/rt2/enabled``
      register the RT 2.0 renderer at boot so the mode is actually available.
    - ``/rtx-transient/dlssg/enabled=false`` pins frame generation off (the
      explicit "1x, no FPS multiplier" decision — see module docstring).
    """
    return {
        "/rtx/rendermode": RT2_RENDERMODE_TOKEN,
        "/rtx-transient/rt2Enabled": True,
        "/persistent/rtx/modes/rt2/enabled": True,
        "/rtx-transient/dlssg/enabled": False,
    }


def renderer_kit_setting_overrides(renderer: str) -> "dict[str, object]":
    """Return the Kit setting overrides for ``renderer`` (empty for legacy)."""
    _validate(renderer)
    if renderer == RENDERER_RT2:
        return _rt2_kit_setting_overrides()
    return {}


def renderer_rendering_mode(renderer: str) -> "str | None":
    """Return the IsaacLab ``--rendering_mode`` preset, or None to leave default."""
    _validate(renderer)
    if renderer == RENDERER_RT2:
        return RT2_RENDERING_MODE
    return None


def format_kit_setting_args(overrides: "dict[str, object]") -> str:
    """Render carb overrides as a Kit CLI arg string: ``--/path=value ...``.

    Booleans are emitted lowercase (``true`` / ``false``) to match Kit's
    setting-parse expectations; everything else is stringified as-is.
    """
    parts: list[str] = []
    for path, value in overrides.items():
        if isinstance(value, bool):
            token = "true" if value else "false"
        else:
            token = str(value)
        parts.append(f"--{path}={token}")
    return " ".join(parts)


def describe_renderer(renderer: str) -> str:
    """One-line human description of the selected renderer for a startup log."""
    _validate(renderer)
    if renderer == RENDERER_RT2:
        return (
            f"RTX Real-Time 2.0 ({RT2_RENDERMODE_TOKEN}), "
            f"{RT2_RENDERING_MODE} preset, frame-generation OFF (1x)"
        )
    return f"legacy RTX Real-Time 1.0 ({LEGACY_RENDERMODE_TOKEN}), app-default preset"


def _validate(renderer: str) -> None:
    if renderer not in RENDERER_CHOICES:
        raise ValueError(
            f"unknown renderer {renderer!r}; expected one of {RENDERER_CHOICES}"
        )
