# Phase 7: Procedural Primitive Room Generation with GPU BFS Solvability

## Context

PPO convergence requires 64-256+ parallel environments. The Infinigen-based pipeline (Phase 6, now `StraferSceneCfg_Infinigen`) maxes out at ~8 envs due to VRAM. Meanwhile, the existing `StraferSceneCfg` uses only 8 hardcoded 0.3m cuboid obstacles on a flat plane — useful for basic obstacle avoidance but lacking the spatial structure (walls, doorways, furniture) needed to learn indoor navigation.

**Solution:** A new `StraferSceneCfg_ProcRoom` that procedurally generates room-like environments from lightweight primitive shapes at each episode reset, with GPU-parallel BFS to guarantee every layout is solvable. Primitive geometry is cheap enough for 256+ envs while providing the spatial diversity the flat-ground setup lacks.

**Theory:** A depth-based policy learns spatial reasoning (clearance, occupancy, path planning), not object identity. Diverse primitive configurations produce the same depth patterns as real furniture. The Infinigen pipeline then becomes the visual-fidelity layer for VLM pretraining and sim-to-real fine-tuning.

---

## 0. Prerequisite Refactors

Before introducing ProcRoom, clean up existing scene infrastructure:

### 0a. Remove hardcoded obstacles from `StraferSceneCfg` / `StraferSceneCfg_NoCam`

The 8 hardcoded `obstacle_0..7` `RigidObjectCfg` entries and their `filter_prim_paths_expr` contact sensor are no longer needed — ProcRoom replaces them entirely. Remove:
- `OBSTACLE_CFG` template and `NUM_OBSTACLES` constant
- `obstacle_0..7` fields from `StraferSceneCfg` and `StraferSceneCfg_NoCam`
- `filter_prim_paths_expr` arrays from both scene configs' contact sensors (switch to `net_forces_w` like ProcScene/ProcRoom)
- `_OBSTACLE_NAMES`, `_RANDOMIZE_OBSTACLES` event term
- `randomize_obstacles` references in `EventsCfg_Ideal/Realistic/Robust`
- `ObstacleCurriculum` from `CurriculumCfg` (obstacle_difficulty term)
- `collision_penalty` / `collision_sustained_penalty` (force_matrix_w variants) from `RewardsCfg` — replace with `collision_penalty_net` / `collision_sustained_penalty_net`

**Impact:** All 18 existing non-ProcScene env configs become obstacle-free flat-ground environments focused on goal navigation + domain randomization. ProcRoom configs replace them as the obstacle avoidance training target.

### 0b. Rename ProcScene → Infinigen

Rename all ProcScene-related symbols to clearly distinguish from the new ProcRoom:

| Old Name | New Name |
|----------|----------|
| `StraferSceneCfg_ProcScene` | `StraferSceneCfg_Infinigen` |
| `CommandsCfg_ProcScene` | `CommandsCfg_Infinigen` |
| `RewardsCfg_ProcScene` | `RewardsCfg_Infinigen` |
| `EventsCfg_ProcScene_Realistic` | `EventsCfg_Infinigen_Realistic` |
| `EventsCfg_ProcScene_Robust` | `EventsCfg_Infinigen_Robust` |
| `CurriculumCfg_ProcScene` | `CurriculumCfg_Infinigen` |
| `StraferNavEnvCfg_Real_ProcDepth` | `StraferNavEnvCfg_Real_InfinigenDepth` |
| `StraferNavEnvCfg_Robust_ProcDepth` | `StraferNavEnvCfg_Robust_InfinigenDepth` |
| `Isaac-Strafer-Nav-Real-ProcDepth-v0` | `Isaac-Strafer-Nav-Real-InfinigenDepth-v0` |
| `Isaac-Strafer-Nav-Robust-ProcDepth-v0` | `Isaac-Strafer-Nav-Robust-InfinigenDepth-v0` |

**Files affected (11):**
- `strafer_env_cfg.py` (24 ProcScene + 8 ProcDepth references)
- `__init__.py` (10 ProcDepth gym registrations)
- `mdp/commands.py` (3 ProcScene comments)
- `mdp/rewards.py` (1 ProcScene comment)
- `test/rewards/test_collision_rewards_procscene.py` → rename to `test_collision_rewards_infinigen.py` (5 refs)
- `test/env/test_env_registration.py` (8 ProcDepth expected env IDs)
- `Scripts/train_strafer_navigation.py` (2 ProcDepth)
- `Scripts/scenegen/prep_room_usds.py` (1 ProcDepth comment)
- `docs/PHASE_6_SYNTHETIC_SCENE_GENERATION.md` (12 ProcDepth)
- `docs/PHASE_7_PROCEDURAL_ROOMS.md` (update after creating)
- `docs/CLAUDE_SEED_PROMPT.md` (5 mixed)

---

## 1. Architecture Overview

```
StraferSceneCfg (existing)     StraferSceneCfg_ProcRoom (new)
─────────────────────────      ──────────────────────────────
No obstacles (removed)         1 RigidObjectCollectionCfg (44 objects)
                                 wall_long_0..7, wall_med_0..7, wall_short_0..3
                                 furn_table_0..1, furn_shelf_0..1, etc.
                                 clutter: boxes, cylinders, spheres, cones, capsules
flat ground + DR only          EventTerm: generate_proc_room
                                 build walls → place furniture → scatter clutter → BFS check
CurriculumCfg:                 CurriculumCfg_ProcRoom:
  goal_distance only             goal_distance + room_complexity
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Object management | `RigidObjectCollectionCfg` | Single `physx.RigidBodyView` for all 44 objects. One batched `write_object_link_pose_to_sim()` call per reset instead of 44 individual writes. |
| Shape diversity | Fixed size categories, not runtime resize | USD prim scale cannot change at runtime (`randomize_rigid_body_scale` is "usd" mode only). Discrete size palette is sufficient — diversity comes from placement, not shape. |
| `replicate_physics` | `True` | All envs have identical shape types (same CuboidCfg/CylinderCfg sizes). Only positions differ at reset → homogeneous per-env assets. |
| Contact detection | `net_forces_w` (no filter) | Same as Infinigen. body_link is wheel-suspended ~10cm above ground; net forces detect wall/furniture collisions, not ground contact. Reuses existing `collision_penalty_net` rewards. |
| Solvability | GPU BFS with robot-radius inflation | Parallel flood-fill across all resetting envs. Conservative AABB rasterization + morphological dilation. |

---

## 2. Primitive Palette (36 objects)

All kinematic (`kinematic_enabled=True`), placed with bottom surface at floor level (center Z = height/2).

### Category A: Wall Segments (20)

| Slot | Shape | Size (X × Y × Z) | Count | Color | Purpose |
|------|-------|-------------------|-------|-------|---------|
| `wall_long_{0-7}` | CuboidCfg | 2.0 × 0.15 × 1.0 | 8 | (0.75, 0.75, 0.75) | Perimeter walls |
| `wall_med_{0-7}` | CuboidCfg | 1.0 × 0.15 × 1.0 | 8 | (0.70, 0.70, 0.72) | Internal walls, perimeter filler |
| `wall_short_{0-3}` | CuboidCfg | 0.5 × 0.15 × 1.0 | 4 | (0.65, 0.65, 0.68) | Doorway framing |

**Budget reasoning:** A 7m perimeter side = 3 long + 1 med. 4 sides × 4 = 16 for perimeter, leaving 4 for internal walls/filler.

### Category B: Furniture (8)

| Slot | Shape | Size (X × Y × Z) | Count | Color | Purpose |
|------|-------|-------------------|-------|-------|---------|
| `furn_table_{0-1}` | CuboidCfg | 0.8 × 0.6 × 0.4 | 2 | (0.55, 0.35, 0.15) | Tables |
| `furn_shelf_{0-1}` | CuboidCfg | 1.2 × 0.3 × 0.8 | 2 | (0.50, 0.30, 0.12) | Bookshelves |
| `furn_cabinet_{0-1}` | CuboidCfg | 0.5 × 0.5 × 0.6 | 2 | (0.45, 0.28, 0.10) | Cabinets |
| `furn_couch_{0-1}` | CuboidCfg | 1.4 × 0.6 × 0.35 | 2 | (0.40, 0.25, 0.55) | Couches |

### Category C: Clutter (16)

| Slot | Shape | Size / Params | Count | Color | Purpose |
|------|-------|---------------|-------|-------|---------|
| `clutter_box_{0-3}` | CuboidCfg | 0.3 × 0.3 × 0.3 | 4 | (0.6, 0.2, 0.2) | Small boxes |
| `clutter_cyl_{0-1}` | CylinderCfg | r=0.15, h=0.4 | 2 | (0.2, 0.5, 0.2) | Trash cans |
| `clutter_flat_{0-1}` | CuboidCfg | 0.4 × 0.4 × 0.15 | 2 | (0.5, 0.5, 0.2) | Floor mats |
| `clutter_sphere_{0-1}` | SphereCfg | r=0.15 | 2 | (0.7, 0.4, 0.1) | Balls, round objects |
| `clutter_cone_{0-1}` | ConeCfg | r=0.12, h=0.35 | 2 | (0.8, 0.5, 0.1) | Traffic cones, markers |
| `clutter_capsule_{0-1}` | CapsuleCfg | r=0.1, h=0.4 | 2 | (0.3, 0.3, 0.6) | Bottles, rolled mats |
| `clutter_tall_cyl_{0-1}` | CylinderCfg | r=0.1, h=0.7 | 2 | (0.4, 0.6, 0.3) | Floor lamps, stands |

**Total: 44 objects per env** (20 walls + 8 furniture + 16 clutter).

**VRAM at scale:** 44 objects × 256 envs = 11,264 simple collision shapes. Compare: the old obstacle setup was 8 × 4,096 = 32,768. ProcRoom is lighter.

Inactive objects parked at `(100, 100, -10)` following existing `ObstacleCurriculum` pattern.

**Shape diversity rationale:** Depth sensors see geometry profiles — spheres, cones, and capsules produce distinct depth silhouettes that cuboids alone cannot represent. Including all five Isaac Lab primitive types ensures the policy encounters the full range of geometric profiles it will see in real environments.

---

## 3. Room Generation Algorithm (`generate_proc_room`)

Runs as an `EventTerm(mode="reset")`. Defined BEFORE `reset_robot` in EventsCfg to ensure spawn points are available when the robot resets.

### Phase 1: Parameter Sampling

Per resetting env (B = batch size):
- `room_w` ~ Uniform(4.0, 7.0) meters
- `room_h` ~ Uniform(4.0, 7.0) meters
- `num_doorways` ~ randint(min_doorways, max_doorways+1)
- `doorway_sides` ~ random subset of {N, S, E, W}
- `doorway_positions` ~ Uniform along each chosen wall
- `doorway_width` ~ Uniform(0.8, 1.2)
- `num_internal_walls` ~ 0 to `max_internal_walls` (from curriculum)
- `num_furniture` ~ 0 to `max_furniture` (from curriculum)
- `num_clutter` ~ 0 to `max_clutter` (from curriculum)

### Phase 2: Wall Construction

Room centered at env origin (0, 0). For each perimeter side:

1. Compute total wall length for this side
2. If doorway on this side: split into left section + gap + right section
3. Greedy-pack each section with wall segments (longest first: 2.0m → 1.0m → 0.5m)
4. Position segments end-to-end along the wall, rotated 0° (horizontal) or 90° (vertical)

Wall segment pose: `pos = (cx, cy, height/2)`, `quat = yaw_to_quat(0 or π/2)`

**Internal walls** (when curriculum enables): Place a wall spanning part of the room width/height, with a doorway gap. Uses `wall_med` segments from the remaining budget.

### Phase 3: Furniture Placement

For each active furniture slot:
1. Choose a random perimeter wall to align against
2. Sample position along the wall (min 0.5m from doorways, min 0.5m from other furniture)
3. Offset from wall by `furniture_depth/2 + wall_thickness/2`
4. Random 0/180° rotation (facing in/out from wall)
5. Rejection sampling: 10 attempts, unplaced furniture gets parked

### Phase 4: Clutter Scatter

For each active clutter slot:
1. Sample random XY within room interior (0.5m inset from walls)
2. Min 0.4m from furniture, min 0.3m from other clutter
3. Random yaw rotation
4. Rejection sampling: 10 attempts, unplaced clutter gets parked

### Phase 5: Pose Assembly and Batched Write

```python
# Assemble all 44 object poses: (B, 44, 7)
poses = torch.zeros(B, 44, 7, device=device)
# ... fill from phases 2-4, park unused slots ...

# Offset by env_origins
env_origins = env.scene.env_origins[env_ids]
poses[:, :, :3] += env_origins.unsqueeze(1)

# Single batched write
collection = env.scene["room_primitives"]
collection.write_object_link_pose_to_sim(poses, env_ids, all_object_ids)
```

### Phase 6: BFS Solvability Check

If BFS fails for any env, remove clutter/furniture and retry (see Section 4.4).

### Phase 7: Extract Spawn Points

Sample K=200 random reachable cells per env, convert to world XY, store on `env._proc_room_spawn_pts`.

---

## 4. GPU BFS Solvability Checker

### 4.1 Occupancy Grid Construction

Grid parameters:
- Resolution: 0.1m per cell
- Max grid: 80 x 80 (covers 8m x 8m room with margin)
- Grid origin: `(-room_w/2 - 0.5, -room_h/2 - 0.5)` per env

For each active object, compute axis-aligned bounding box after rotation:
```
hx = |wx/2 * cos(theta)| + |wy/2 * sin(theta)|
hy = |wx/2 * sin(theta)| + |wy/2 * cos(theta)|
AABB = [cx +/- hx, cy +/- hy]
```
Convert to grid cell ranges, mark occupied.

### 4.2 Robot Radius Inflation

Dilate the occupancy grid by the robot's half-width using `F.max_pool2d`:

```python
R = ceil(0.28 / 0.1)  # = 3 cells
kernel = 2*R + 1       # = 7
inflated = F.max_pool2d(occupancy.unsqueeze(1), kernel_size=kernel, stride=1, padding=R)
free_space = (inflated < 0.5)  # (B, Gx, Gy) bool
```

### 4.3 Parallel Wavefront BFS

```python
def _gpu_bfs(free_space, start_cells, max_iterations=200):
    """Parallel BFS via iterative morphological dilation.

    Args:
        free_space: (B, Gx, Gy) bool -- True = passable
        start_cells: (B, 2) int -- (row, col) of BFS start

    Returns:
        reachable: (B, Gx, Gy) bool
    """
    B, Gx, Gy = free_space.shape
    reachable = torch.zeros(B, 1, Gx, Gy, device=free_space.device)
    # Seed start positions
    reachable[torch.arange(B), 0, start_cells[:, 0], start_cells[:, 1]] = 1.0
    free = free_space.unsqueeze(1).float()

    for _ in range(max_iterations):
        expanded = F.max_pool2d(reachable, kernel_size=3, stride=1, padding=1)
        new_cells = expanded * free * (1.0 - reachable)
        if new_cells.sum() == 0:
            break
        reachable = (reachable + new_cells).clamp(max=1.0)

    return reachable.squeeze(1) > 0.5
```

**Performance:** 80x80 grid, B=256 envs = ~160 worst-case iterations, each a single `F.max_pool2d` CUDA kernel on 256x1x80x80 tensor (~5 MB). Total: well under 10ms per reset batch.

### 4.4 Retry on BFS Failure

After BFS from room center:
1. Count reachable free cells. Need at least `min_reachable_cells` (e.g., 100).
2. For envs that fail:
   - **Retry 1-3:** Park one clutter object, rebuild grid, re-run BFS
   - **Retry 4:** Park one furniture object
   - **Retry 5:** Park internal walls
   - **Fallback:** Accept room as walls-only (always solvable for a single rectangular room with doorways)

### 4.5 Spawn Point Extraction

From the reachable mask, sample K=200 random reachable cells per env. Convert grid coords to world XY coords. Store as `(num_envs, K, 2)` tensor on the env.

Used by both `reset_robot_proc_room` and `GoalCommandProcRoom`.

---

## 5. Robot Reset Event: `reset_robot_proc_room`

Mirrors `reset_robot_state_on_floor` (used by Infinigen) but reads from dynamic per-env spawn points computed by BFS:
- Indexes into `env._proc_room_spawn_pts[env_ids]`
- Random index bounded by `env._proc_room_spawn_count[env_ids]`
- Random yaw, fixed Z=0.1, quaternion construction
- `robot.write_root_state_to_sim()`

---

## 6. Goal Command: `GoalCommandProcRoom`

Subclass of `GoalCommand`. Overrides `_resample_command` to sample goals from the dynamic per-env BFS reachable points (`env._proc_room_spawn_pts`) instead of static config-time points or uniform box sampling. Same rejection-sampling pattern (min distance from robot, 10 attempts, fallback at min_distance in random direction).

---

## 7. Room Complexity Curriculum

### `RoomComplexityCurriculum`

Follows `ObstacleCurriculum` pattern: tracks consecutive goal-reach successes per env, advances difficulty level.

| Level | Internal Walls | Furniture | Clutter | Description |
|-------|---------------|-----------|---------|-------------|
| 0 | 0 | 0 | 0 | Empty rectangular room |
| 1 | 0 | 2 | 0 | Sparse furniture |
| 2 | 0 | 4 | 4 | Moderate obstacles |
| 3 | 1 | 4 | 8 | Internal wall added |
| 4 | 1 | 6 | 12 | Dense clutter |
| 5 | 2 | 8 | 16 | Full complexity |

Stores `env._proc_room_difficulty` per env. `generate_proc_room` reads this to determine `max_internal_walls`, `max_furniture`, `max_clutter`.

---

## 8. Environment Configs

### Scene Configs

```
StraferSceneCfg_ProcRoom          -- with camera (depth training)
StraferSceneCfg_ProcRoom_NoCam    -- without camera (fastest training)
```

Both use:
- `room_primitives: RigidObjectCollectionCfg` (44 objects from `_build_proc_room_palette()`)
- `contact_sensor` with no filter (net_forces_w)
- `env_spacing=10.0` (rooms up to 7m + margin)

### Registered Gym Environments (8 new = 4 train x 2 play)

| Gym ID | Realism | Camera | Train num_envs |
|--------|---------|--------|----------------|
| `Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0` | Realistic | No | 256 |
| `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` | Realistic | Depth | 64 |
| `Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0` | Robust | No | 256 |
| `Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0` | Robust | Depth | 64 |

Each with `-Play-v0` variant (num_envs=50 NoCam, 8 Depth).

Reuses: `RewardsCfg_Infinigen` (collision_penalty_net), `TerminationsCfg`, Action configs.

---

## 9. File Changes

### Prerequisite Refactors (Section 0)

| File | Changes |
|------|---------|
| `strafer_env_cfg.py` | Remove `OBSTACLE_CFG`, `NUM_OBSTACLES`, `obstacle_0..7` from `StraferSceneCfg`/`StraferSceneCfg_NoCam`, switch contact sensors to no-filter, remove `_OBSTACLE_NAMES`, `_RANDOMIZE_OBSTACLES`. Switch `RewardsCfg` to use `collision_penalty_net`. Remove obstacle_difficulty from `CurriculumCfg`. Rename all ProcScene→Infinigen symbols. |
| `__init__.py` | Rename ProcDepth→InfinigenDepth gym registration IDs (4 entries). |
| `mdp/commands.py` | Update ProcScene→Infinigen comments. |
| `mdp/rewards.py` | Update ProcScene→Infinigen comment. |
| `test/rewards/test_collision_rewards_procscene.py` | Rename file to `test_collision_rewards_infinigen.py`, update internal references. |
| `test/env/test_env_registration.py` | Update expected env IDs (ProcDepth→InfinigenDepth), update tier counts, add ProcRoom env IDs. |
| `Scripts/train_strafer_navigation.py` | Update ProcDepth→InfinigenDepth references. |
| `Scripts/scenegen/prep_room_usds.py` | Update ProcDepth→InfinigenDepth comment. |
| `docs/PHASE_6_SYNTHETIC_SCENE_GENERATION.md` | Update ProcDepth→InfinigenDepth references. |
| `docs/CLAUDE_SEED_PROMPT.md` | Update ProcScene/ProcDepth references. |

### New Files
| File | Contents |
|------|----------|
| `mdp/proc_room.py` | `_build_proc_room_palette()`, `generate_proc_room()`, `_build_occupancy_grid()`, `_inflate_obstacles()`, `_gpu_bfs()`, `_extract_spawn_points()` |
| `test/events/test_proc_room.py` | Test room generation: walls form perimeter, BFS validates solvability, spawn points extracted, inactive objects parked |
| `test/commands/test_goal_proc_room.py` | Test `GoalCommandProcRoom` samples from dynamic BFS points, min distance, fallback |
| `test/curriculums/test_room_curriculum.py` | Test `RoomComplexityCurriculum` level progression |

### Modified Files (ProcRoom additions)
| File | Changes |
|------|---------|
| `strafer_env_cfg.py` | Add `StraferSceneCfg_ProcRoom`, `StraferSceneCfg_ProcRoom_NoCam`, `CommandsCfg_ProcRoom`, `EventsCfg_ProcRoom_Realistic/Robust`, `CurriculumCfg_ProcRoom`, 4 train + 4 play env configs. Import `RigidObjectCollectionCfg`. |
| `mdp/events.py` | Add `reset_robot_proc_room()` |
| `mdp/commands.py` | Add `GoalCommandProcRoom`, `GoalCommandProcRoomCfg` |
| `mdp/curriculums.py` | Add `RoomComplexityCurriculum` |
| `mdp/__init__.py` | Export new symbols |
| `__init__.py` | Register 8 new ProcRoom gym environments |

---

## 10. Verification Checklist

1. **Prerequisite refactors:** ProcScene→Infinigen rename compiles, all 18 existing envs still register and load (no obstacles)
2. **Scene creation:** 256 ProcRoom envs load without errors, all 44 primitives visible per env
3. **Room generation:** Walls form closed perimeters with doorway gaps
4. **BFS validation:** Every room has a connected reachable region from center
5. **Robot spawn:** Robot always appears inside the room on the floor
6. **Goal placement:** Goals always sample from reachable positions
7. **Collision detection:** `net_forces_w` fires when robot contacts walls/furniture/clutter
8. **Curriculum:** Room complexity starts at level 0, advances with consecutive successes
9. **Performance:** >100 FPS at 256 envs NoCam; BFS overhead <10ms per reset batch
10. **VRAM:** Well within budget (primitive shapes, no heavy meshes)
11. **Tests pass:** All existing tests pass after refactor; new ProcRoom tests pass
