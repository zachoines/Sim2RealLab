# Navigation env-cfg composition contract

Strafer navigation environments are **composed over four orthogonal axes** —
sensor stack × scene source × realism × objective — not hand-written one class
per matrix cell. A new variant is a composition, not a subclass. The
authoritative in-code statement is the
[`composed_env_cfg.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)
module docstring; the registration table is
[`tasks/navigation/__init__.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py).
The shared MDP building blocks the composition selects among (per-realism
observation / action / event cfgs, scene cfgs, commands / rewards /
terminations / curriculum) live in
[`strafer_env_cfg.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py).

## The four axes

Set on a variant; `_ComposedStraferNavEnvCfg.__post_init__` materializes the
standard manager cfgs from them.

- **`SensorStackCfg(cameras_required=...)`** — tuple over `rgb_full` /
  `depth_full` (640×360 perception camera) and `rgb_policy` / `depth_policy`
  (80×60 policy camera). Drives which camera prims render **and** which image
  observation terms the policy gets. An empty tuple is camera-free.
- **`SceneSourceCfg(kind=...)`** — `plane` / `infinigen` / `procroom` / `none`.
  The seam that makes a foreign USD a parameter, not a subclass.
- **`RealismCfg(level=...)`** — `ideal` / `real` / `robust` DR + noise tier.
- **`ObjectiveCfg(kind=...)`** — `goal` (fixed goal pose per episode) or
  `subgoal` (rolling subgoal along a sim-planned path). Selects the command /
  reward / termination blocks; observations are objective-agnostic because
  the goal-shaped obs terms read whatever command term is registered under
  `goal_command`. `subgoal` is composed for the `procroom` source only — its
  planner consumes the occupancy grids the procedural-room generator builds.

## Gym-ID scheme

Two families, prefix `Isaac-Strafer-Nav-`:

- **RL** (`-RLDepth-Real-v0`, `-RLDepth-Robust-v0`, `-RLNoCam-v0`,
  `-RLNoCam-Subgoal-Real-v0`, `-RLNoCam-Subgoal-Robust-v0`, each + `-Play`):
  **fixed** sensor stack — the obs contract a trained policy was fitted against.
- **Capture** (`-Capture-Teleop-v0`, `-Capture-Bridge-v0`, `-Capture-Coverage-v0`):
  the registered stack is a **default preset**, operator-overridable per session.

The runner cfg pairs with the observation profile: a depth-image obs uses the
CNN depth runner, a proprioceptive obs the MLP runner. Old per-cell gym IDs
(`-Real-v0`, `-Real-Depth-v0`, `-Real-ProcRoom-Depth-v0`, `-InfinigenPerception-Play-v0`, …)
are retired — the old→new table is in
[`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md).

## The two invariants

1. **RL obs/action contract is preserved.** A composed RL variant must emit a
   byte-identical observation tensor, action layout, and DR sequence to the
   config a checkpoint trained against — a subtly different tensor breaks the
   DEPTH inference package and any checkpoint silently. The guard is the
   frozen-golden-hash gate in
   [`test_sim/env/test_composition_contract.py`](../../../source/strafer_lab/test_sim/env/test_composition_contract.py);
   the depth observation a checkpoint consumes is also pinned by
   [`recurrent-policy-contract.md`](recurrent-policy-contract.md). A camera's
   rendered channels are excluded from the gate — a depth-only obs never reads
   RGB, so what a camera renders beyond the observed channels does not move the
   hash. This is **not** license to drop RGB: the policy camera must still
   render `rgb` for the RTX viewport / `--video` colour pipeline (no rgb render
   product → black clips and a frozen headed viewport), enforced by
   `_prune_scene_cameras` and a check in the same contract test.
2. **The capture sensor stack and the writer schema are one parameter.**
   `cameras_required` drives both the rendered cameras and
   [`lerobot_writer.build_features`](../../../source/strafer_lab/strafer_lab/tools/lerobot_writer.py)
   — absent modalities produce **no columns** (never zero-filled), and
   `add_frame` validates each frame against the declared stack.
   [`capture.py --sensors`](../../../source/strafer_lab/scripts/capture.py) (preset or token list)
   resolves to that tuple and applies it to `env_cfg.sensors` before
   `gym.make`, so the env renders exactly what the writer records.

## Adding a variant

Compose over the axes in `composed_env_cfg.py` and register in `__init__.py`.
If it is policy-facing (a checkpoint will train against it), its sensor stack
is **fixed** and it is snapshot-gated — add a frozen golden hash. Capture
variants are not snapshot-gated: their gate is "renders exactly the intended
stack; writer declares matching columns."
