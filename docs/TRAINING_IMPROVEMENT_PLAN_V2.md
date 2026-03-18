# Training Improvement Plan V2

**Date:** 2026-03-18
**Status:** Planned
**Context:** CNN-GRU-MLP depth policy with DAPG, ProcRoom environments

## Problem Summary

Training run at ~800 iterations revealed several issues:

1. **Adaptive LR crashed to 1e-5 by step ~100** — DAPG's strong early gradient
   caused high KL divergence, triggering the adaptive controller to throttle
   LR to its floor. This effectively killed RL learning.
2. **DAPG NLL converges too fast** — with ~60 demo episodes the policy
   memorizes the demo buffer quickly. The demo distribution is too narrow.
3. **Entropy collapses (0.75 -> 0.3)** — DAPG shrinks action variance to
   match expert actions. The entropy bonus can't push back at 1e-5 LR.
4. **Curriculum creates distribution mismatch** — demos sample random
   difficulty levels, but curriculum starts all envs at level 0. The policy
   trains on easy layouts but demos show hard ones.
5. **Demo collection is slow/difficult** — robot-relative gamepad control
   from 3rd-person view causes errors near obstacles, producing imperfect
   demos.

## Implementation Tasks

### Task 1: Fix Learning Rate (Critical)

**File:** `source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py`

Switch depth config from `schedule="adaptive"` to `schedule="fixed"` with
`learning_rate=3.0e-4`. The adaptive controller is counterproductive with
DAPG — the strong initial BC gradient looks like a dangerously large policy
update, causing the controller to kill LR before RL even starts.

Also increase `entropy_coef` from 0.01 to 0.02 to counteract DAPG's
variance-collapsing effect. With fixed LR the entropy bonus will actually
have effect.

### Task 2: Disable Curriculum, Enable Random Room Sampling

**Files:**
- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`
  (ProcRoom env configs)

Remove `GoalDistanceCurriculum` and `RoomComplexityCurriculum` from ProcRoom
env configs. Instead:

- Set `_proc_room_difficulty` to sample uniformly from [0, max_level] per env
  on each episode reset (via an event function).
- Set goal distance range to full range from the start (no curriculum).
- This makes the online training distribution match the demo distribution
  (random difficulty per episode).

The curriculum classes can remain in the codebase for future use — we just
remove them from the active env configs.

### Task 3: World-Frame Gamepad Controller

**File:** `source/strafer_lab/scripts/collect_demos.py`

Replace robot-relative twin-stick control with world-frame control:

- **Left stick** = world-frame velocity direction
  - Stick angle maps to world-frame movement direction
  - Stick magnitude maps to speed (0 to max)
  - e.g., stick pointing screen-right = robot moves world +X regardless of
    robot heading
- **Right stick X** = world-frame heading target
  - Stick angle sets desired absolute heading (not rotation rate)
  - When centered, heading holds current value
  - This allows spinning in circles while moving in straight lines

The conversion from world-frame to robot-frame commands happens in the
collection script (not the env):

```python
# World-frame desired velocity
world_vx = stick_magnitude * cos(stick_angle)
world_vy = stick_magnitude * sin(stick_angle)

# Transform to robot body frame using current heading
cos_h = cos(robot_heading)
sin_h = sin(robot_heading)
body_vx =  cos_h * world_vx + sin_h * world_vy
body_vy = -sin_h * world_vx + cos_h * world_vy

# Heading controller: P-controller toward desired heading
heading_error = wrap_angle(desired_heading - robot_heading)
omega = kp * heading_error  # clamped to [-1, 1]
```

The action sent to the env is still `[body_vx, body_vy, omega]` — the env
interface doesn't change, only the human input mapping.

**Benefits:**
- Eliminates mental rotation when teleoperating from overhead view
- Much easier precision control near obstacles
- Decoupled translation and rotation = cleaner demo trajectories
- Should significantly reduce demo collection errors and time

**Reading robot heading:** The script needs the robot's current yaw to do the
world-to-body transform. This is available from the env's observation vector
(heading components sin/cos) or directly from
`env.unwrapped.scene["robot"].data.root_quat_w`.

### Task 4: Improve DAPG Robustness with Limited Demos

**File:** `source/strafer_lab/strafer_lab/tasks/navigation/agents/bc_loss.py`

With ~60 episodes the policy memorizes demos quickly. Mitigations:

1. **Action noise injection**: Add small Gaussian noise (std=0.05) to demo
   actions during sampling. This regularizes the BC target and prevents
   the policy from overfitting to exact gamepad inputs (which are noisy
   themselves).

2. **Reduce BC batch size**: From 256 to 128. With a small buffer, sampling
   256 per mini-batch means high overlap between batches within one update
   cycle. Smaller batches = more variance = slower memorization.

3. **Slower initial weight**: Start `bc_weight` at 0.03 instead of 0.05.
   The DAPG term is inside the loss and gets amplified by 5 learning epochs
   x 4 mini-batches = 20 gradient steps per update call.

### Task 5: Collect More Demos with World-Frame Controller

After implementing Task 3, collect a new larger demo set:

- Target: 150+ episodes across difficulty levels 0-7
- Use `--max_difficulty 7` to randomly sample room complexity
- With world-frame control, collection should be ~2x faster and produce
  higher quality trajectories
- Store rewards in HDF5 (already implemented) for future return-based
  filtering

## Hyperparameter Summary

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `schedule` | adaptive | **fixed** | Adaptive LR killed training |
| `learning_rate` | 3e-4 (crashed to 1e-5) | **3e-4 (fixed)** | Stable throughout training |
| `entropy_coef` | 0.01 | **0.02** | Counteract DAPG variance collapse |
| `bc_weight` | 0.05 | **0.03** | Slower memorization of small demo set |
| `bc_batch_size` | 256 | **128** | Less overlap with small buffer |
| curriculum | GoalDistance + RoomComplexity | **disabled** | Match demo distribution |
| room difficulty | curriculum 0->7 | **random uniform [0, max]** | Consistent with demos |
| demo action noise | none | **N(0, 0.05)** | Regularize BC target |
| gamepad control | robot-relative | **world-frame** | Easier teleop from 3rd person |

## Execution Order

1. Task 1 (fix LR + entropy) — immediate, config change only
2. Task 2 (disable curriculum) — requires env config edits
3. Task 4 (DAPG robustness) — bc_loss.py tweaks
4. Task 3 (world-frame controller) — collect_demos.py rewrite
5. Task 5 (collect more demos) — manual collection with new controller
6. Retrain with all improvements

Tasks 1-4 can be implemented in parallel. Task 5 depends on Task 3.
