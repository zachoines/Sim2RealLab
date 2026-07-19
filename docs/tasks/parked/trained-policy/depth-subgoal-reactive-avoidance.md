# Re-enable the DEPTH_SUBGOAL depth penalty in a hardened ProcRoom (reactive obstacle avoidance)

**Type:** task / training-env hardening + reward re-enable
**Owner:** DGX (`strafer_lab` lane — env generation + reward economics + a DEPTH-rate training run)
**Priority:** P3 — closes the *reactive-avoidance* half of [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md), whose depth **tracking** half shipped in PR #138 with the penalty inert.
**Estimate:** L (env authoring is the bulk; the training recipe is a warm-start, not a fresh convergence)
**Branch:** task/depth-subgoal-reactive-avoidance

## Un-park trigger

Parked until **both**:

1. [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md)'s depth-tracking checkpoint is accepted (the warm-start base this brief resumes from), and
2. the operator is ready to invest a DEPTH-rate training run in reactive avoidance (it contends with other GPU work on the GB10).

The **value is confirmed** (operator, 2026-07-06), so this is a real planned follow-on, not a speculative filing: Nav2 pre-clears static geometry, but it updates the costmap *slowly* for dynamic/late-arriving obstacles and cannot give a blind (proprioceptive) agent abundant clearance in tight passages (doorways). The depth penalty earns its place on exactly those two cases — **dynamic obstacles + tight clearances** — neither of which the current ProcRoom env presents.

## Why the penalty ships inert (established — do not re-litigate)

PR #138 ships `depth_obstacle_proximity` at **weight 0.0** (term kept wired; re-enable is a one-float flip). The diagnostic arc (three DEPTH_SUBGOAL training runs) proved the penalty is *starved*, not broken, and pinned the causes in code:

1. **Paths are pre-cleared straight shots.** The A* planner greedily line-of-sight-shortcuts (`path_planner/planner.py` `_shortcut`) and inflates occupancy by the full robot radius (`mdp/proc_room.py` `_inflate_obstacles`, `ROBOT_HALF_WIDTH=0.28`). Every planned path keeps ≥ 0.28 m off all clutter, so the forward camera rarely faces an on-path obstacle to steer around. Improvement slope was shallow (raw penalty ~0.285 → 0.21); raising the weight −0.25 → −1.0 left the sensed obstacle distance unchanged (~0.43 m) and destabilized late.
2. **The 0.3 m off-path corridor forbids the penalty's own goal.** `_SUBGOAL_MAX_OFF_PATH_M = 0.3 m` drives *both* the `off_path_divergence` termination *and* the −50 one-shot penalty — so deviating far enough to clear an obstacle terminates the episode and pays −50.
3. **~10 % of collisions are blind rear bumps.** `reset_robot_proc_room` spawns at fully random yaw decoupled from the path tangent, so the robot rotates/reverses to start — backing into obstacles the forward-only `d555_camera` cannot see. No forward penalty can touch these.

**"Seed obstacles near the path" as literally proposed does not work** — the radius-inflated A* routes around near-path clutter, `_shortcut` re-straightens, and the room's solvability retry (`proc_room.py` ~779–807) deletes it. The env must be *authored* to force on-path threading, not just made denser (density is already locked at max level 7).

## Approach

### Load-bearing constraint: do not break the NOCAM_SUBGOAL env

The subgoal scene/path generation is **shared**: `generate_proc_room`,
`CommandsCfg_ProcRoom_Subgoal` (the A* + subgoal command), and the
`room_primitives` collection are used by *both* `StraferSceneCfg_ProcRoom` (depth)
and `StraferSceneCfg_ProcRoom_NoCam`. So the hardening below is **not** a set of
cfg overrides on the shared procroom — every path/obstacle/spawn change must land
in a **separate depth-only configuration** (a new scene source, e.g.
`procroom_gauntlet`, and/or a depth-only objective/cfg subclass selected via the
existing `(source, profile)` seam), or behind flags that **default OFF for NOCAM**.

Why this matters beyond byte-identity: for NOCAM_SUBGOAL the obstacles are a
*secondary path-enforcement* signal — the blind policy has no way to sense them,
so straying off the path and colliding is exactly what teaches it to stay on the
path. Curvy paths, near-path gates, and moving obstacles would corrupt that
training signal (and the deployed goal-a NOCAM_SUBGOAL checkpoint's contract).
Keep the NOCAM_SUBGOAL env exactly as it is. *(If preserving a NOCAM-compatible
shared procroom proves to burden the depth work, the fallback — operator's call —
is to split a NOCAM-optimized procroom variant off so depth can evolve freely;
default is separate depth-only variants, NOCAM untouched.)*

### In-scope for the first re-enable — MUST-haves (all in a depth-only variant; NOCAM_SUBGOAL untouched)

- **M1 — authored per-episode near-path gate corridor.** Inside a depth-only ProcRoom variant, after the room is built: stash a start + goal (≥ `min_goal_distance` = 2.0 m) and place 2–4 **tall** gate obstacles (≥ 0.4 m, taller than the ~0.35 m camera) alternating either side of the reference line, each inflated boundary intruding ~0.1 m **across** the centerline — forcing an S-bend A* must thread, opposite side guaranteed open, centers ~0.45–0.6 m off-reference so a tracking robot passes at ~0.3–0.5 m surface (inside the sub-1 m in-FOV gradient band). Make the solvability retry **gate-aware** so it cannot silently delete the corridor; reject background clutter within `corridor_halfwidth + margin` of the reference. This is the doorway/tight-clearance case Nav2 can't give margin in.
- **M2 — widen the DEPTH-ONLY off-path corridor to ~0.6 m** (inflation 0.3 + robot radius 0.28), moving the −50 cliff a full radius past the natural detour band. A depth-only override of `_SUBGOAL_MAX_OFF_PATH_M` that moves **both** the termination and the −50 penalty together (they share the constant). **Keep NOCAM at 0.3 m.** Open decision — see risks — global widen vs gate on a sensed depth obstacle within threshold (the gated form preserves crisp tracking but couples the depth percept into the termination).
- **M3 — rebalance detour economics.** Widening alone leaves the room unused: at a 0.3 m detour the ~−0.25 depth saving barely beats the −2 `cross_track` cost. Reduce `cross_track` (−2 → ~−1) or add a cross-track deadband inside the widened corridor so the detour is net-positive.
- **M4 — a *fair* spawn, not a biased one.** Do **not** bias spawn heading toward the goal — rotating in place to face the path before driving is a valuable skill worth preserving, and the `backward_motion` penalty already teaches "look before you reverse." The ~10 % rear bumps are unfair only because the spawn *isn't clear*: guarantee no obstacle within the robot's **pivot radius + margin** at spawn (extend the existing `randomize_obstacles` `min_robot_dist=0.6` clearance and apply it to the `generate_proc_room` walls/furniture/clutter around the spawn, in the depth-only variant), so a robot that pivots on its axis has **no blind-spot obstacle it couldn't have seen**. Then a rear bump is a genuine policy error (learnable: look-then-move), not an unavoidable start-state artifact — and the collision metric becomes interpretable. Keep the random spawn yaw. (Cheap and orthogonal — could land independently.)

### Staged AFTER the M1–M4 static proof lands — NICE-to-haves

- **N1 — moving obstacles (the dynamic case; highest transfer value).** Scope as a warm-started **YIELD** task: a slow (≤ ~0.4 m/s) obstacle crosses the corridor transiently and the robot slows/stops *in-corridor* (learnable with the GRU's ~1.6 s memory) — **not** steer-around, which trips off_path. Constrain motion so it never permanently seals the sole BFS-guaranteed path (else episodes become unwinnable). Its own sub-phase.
- **N2 — roof plane (containment/lighting realism only).** "Invisible to recording" is achievable **only** as a global session-layer hide while `--video` is active (RTX does not honor per-camera visibility — the known coverage-`--video` gotcha), reusing the existing `(ceiling|roof|…)` hide regex — never simultaneously policy-visible + record-hidden. Keep it **≥ 1.5 m** (nearest ceiling pixel ~2.9 m ≫ the 1.0 m threshold; a lower roof re-taxes the penalty because only *descending* rays are floor-excluded) or add a ceiling-exclusion to the reward. Expect a top-row depth-distribution shift → warm-start.
- **N3 — richer *procedural-primitive* shape variety (AABB-preserving).** The obstacle factory already mixes Cuboid/Cylinder/Sphere/Cone/Capsule with analytic `OBJECT_SIZES` AABBs — **extend it** with more organic and angular primitive shapes (and composite primitives), each carrying a clean AABB the occupancy rasterizer, the BFS solvability check, and the geometric `procroom_obstacle_proximity_penalty` all consume. This enriches the **depth observation's** shape realism for training (better DeFM generalization, a smaller sim2real depth-silhouette gap) without breaking the analytic-AABB contract. Where a genuinely organic shape needs a tighter fit, decouple the layers: a conservative AABB proxy for occupancy/planning/geometric-penalty, the organic mesh only for the render (depth) and physics collider. **NOT via Infinigen** — correcting the earlier note: Infinigen scenes are VRAM-prohibitive and not densely-cluttered enough for obstacle-avoidance *training*; their role is policy *evaluation/validation* and VLM/VLA capture through the Harness. Clustered-obstacle training shape variety comes from ProcRoom primitives. Effort scales with how many shapes; keep the AABB the source of truth for geometry. **Owner pointer (2026-07-19 consult):** the concrete design of this mesh-variety axis and any curated low-poly corpus live in [`procroom-depth-enrichment`](../../active/trained-policy/procroom-depth-enrichment.md) Tier 3; if this brief unparks, it *consumes* that machinery, it does not re-design it.

### Training recipe

Warm-start from the weight-0 depth tracker (never from-scratch — the penalty gradient opposes the bootstrap gradient and re-collapses; never an in-run weight ramp — no reward-weight curriculum plumbing exists, and it would starve without the env fix anyway). Sequence: **(1)** re-baseline penalty-OFF on the NEW env (S-curves lower completion — separate task difficulty from penalty effect); **(2)** enable the depth penalty at a modest **~−0.25** (the operator's −1.0 result destabilizes late and does not lower collisions). 64-env cap (RTX depth-render bound), GB10. Optional sweep: weight {−0.25, −0.5} × corridor {0.5, 0.6, 0.7} × cross-track {−1, deadband} × gate intrusion {0.05, 0.1, 0.15} m.

## Acceptance criteria

What finally earns the penalty its place — all on the NEW hardened env, penalty-ON vs the penalty-OFF re-baseline on the *same* env:

- [ ] Penalty-ON lowers the forward-facing collision rate / raises minimum forward clearance vs the penalty-OFF re-baseline.
- [ ] No completion regression below the re-baseline; no late entropy collapse.
- [ ] The raw obstacle-proximity slope is **non-shallow** (materially steeper than the ~0.285 → 0.21 seen in the flat env).
- [ ] The policy **visibly deviates** within the widened corridor to clear an authored gate (video / screenshots).
- [ ] Rear bumps excluded from the collision metric (M4 landed, or a forward-arc-only counter).
- [ ] (N1, if pursued) The policy yields to a crossing obstacle and resumes — reaches the goal without collision in a ≥ 95 % held-out seeded set.

## Design decisions already made (do not re-open)

Term shipped inert at weight 0.0 (kept wired) in #138; re-enable = warm-start, not from-scratch, not an in-run ramp; modest ~−0.25, not −1.0; **reuse the same 4 DEPTH_SUBGOAL task IDs** (obs/action layout unchanged — the 2×2 matrix is closed; a retrain is expected); **the NOCAM_SUBGOAL env is untouched** — because `generate_proc_room` / `CommandsCfg_ProcRoom_Subgoal` / `room_primitives` are shared and its obstacles are NOCAM's secondary path-enforcement, all hardening lands in a **separate depth-only variant** (new scene source / cfg subclass) or behind NOCAM-default-off flags, never as an in-place edit of the shared procroom; a corridor change moves termination + penalty together; authored gates > raw density; gates taller than the camera; moving obstacles = YIELD; roof = global-hide-during-`--video` only; **spawn = *fair* (clear pivot radius), NOT heading-biased** — random yaw and the rotate-to-face + look-before-reverse skills are preserved; **shape variety = more ProcRoom primitives with clean AABBs**, NOT Infinigen (which is eval/validation/VLA-capture, not training) and NOT AABB-breaking meshes. **The observation vector must not change** — if it does, the depth-obs golden flips and the tracking checkpoint won't warm-start.

## Out of scope / open decisions for the picking-up PR

- **Corridor: global widen vs sensed-obstacle-gated.** Global ~0.6 m weakens the crisp "stay in the cleared corridor" tracking invariant; gating on a sensed depth obstacle preserves tight tracking but couples the depth percept into the termination and adds complexity. Gated is more deployment-faithful. Operator picks the fidelity/complexity trade.
- **Dynamic-first vs static-first.** The transfer value skews toward N1 (dynamic) per the Nav2 rationale, but M1 (static authored gates) is the cheaper, learnable mechanism proof and also covers the tight-clearance/doorway case. Recommended order is M1–M4 then N1; revisit if a dynamic-only scope is preferred.
- **Real-robot rear sensing.** M4 only makes the *training* start fair (a clear pivot radius, so a blind-spot bump is a learnable error not an unfair one). On the real robot the residual blind-spot risk is genuine — its fix is a rear proximity sensor (sim + hardware), which is its own brief. Note the two are complementary: M4 teaches look-then-move, rear sensing removes the blind spot entirely.
