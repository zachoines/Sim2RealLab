# Shared placement and park-order surgery for the ProcRoom generator

**Type:** design consult response + implementation brief (the design is
reviewed here; the code lands in the PR split below)
**Owner:** DGX (`strafer_lab` lane — `mdp/proc_room.py`, env cfgs, the
scratch descriptor instrument, a new numpy-only path-statistics tool)
**Priority:** P2 — opens the operator-ratified enrichment v2 batch, the
closed set of ProcRoom improvements before the single enriched
DEPTH_SUBGOAL v2 retrain. Consumes
[`procroom-depth-enrichment`](procroom-depth-enrichment.md)'s calibration
table and F3 outcome.
**Estimate:** L (guard net + instruments ~1 day; surgery ~1 day; H1 and F2
are separately gated and may not ship at all)
**Branch:** task/procroom-placement-architecture

## Story

As a **DGX operator preparing the enriched DEPTH_SUBGOAL v2 retrain**, I
want **one placement/park-order design that the D4 frequency lever, the D1
rebalance, H1 compounds and F2 internal walls all parameterize**, so that
**each lever is a data change against a proven-untouched vanilla path
instead of a fresh incision into a generator NOCAM depends on forever**.

## Context bundle

- [`procroom-depth-enrichment`](procroom-depth-enrichment.md) — the parent
  brief: the calibration table every gate reads, the F3 outcome that
  triggered this consult, the F2 story constraints, the retirement
  lifecycle.
- [`depth-subgoal-reactive-avoidance`](../../parked/trained-policy/depth-subgoal-reactive-avoidance.md)
  — records why NOCAM's obstacles are load-bearing (they are its secondary
  path-enforcement signal) and owns the M4 spawn-clearance item that
  collides with this brief's D1 lever.
- [`env-composition-contract`](../../context/env-composition-contract.md) —
  the leaf-add vs structural-extension line, and the `enrich_depth` seam
  this design rides unchanged.
- [`path-planning-architecture`](../../context/path-planning-architecture.md)
  — the one shared A* the new path statistics reuse without modification.
- [`branching-and-prs`](../../context/branching-and-prs.md),
  [`conventions`](../../context/conventions.md).

## Context — what the consult established

### The unifying finding

The F3 session localized the D4 residual as *frequency/proximity, not
height*. Measuring further turns that into a single root cause that also
explains two other regressions:

**Enrichment cleared the room interior and pushed the robot away from what
remained.** The perimeter-biased clutter law
(`_ENRICH_CLUTTER_WALL_BIAS = 0.5`), the un-pinned difficulty `U[4,7]`, and
the extra robot-spawn erosion (`_ENRICH_ROBOT_SPAWN_INFLATION = 2`, a 0.2 m
standoff on top of the 0.3 m robot radius) each move geometry away from the
camera. Together they explain, in one mechanism, all of:

| Descriptor | vanilla | enriched | Infinigen | direction |
|---|---|---|---|---|
| D1 near-band median (m) | 1.17 | 1.56 | 1.19 | overshot past the target |
| D4 high-row near (%) | 43.7 | 34.6 (36.4 post-F3) | 40.6–58.9 | moved away from the band |
| path turn density (rad/m, median) | 0.241 | **0.000** | 0.242 (singleroom) | collapsed to zero |
| paths that bend >1.05 tortuosity (%) | 43.7 | 16.5 | 76.7 | roughly halved |
| path clearance below 0.6 m (% of arc) | 71.7 | 21.3 | 72.1 | roughly halved |

> Superseded by PR-A. The depth rows reproduce exactly under the capture
> protocol that produced them, but D4 moves 5–7 points on capture protocol
> alone. The three path rows compare a sealed procedural room against a
> scanned scene the robot was allowed to *leave*; measured identically, the
> singleroom comparator is straighter than the enriched generator, and
> curvature in the corpus is a multi-room property. See
> [What PR-A measured](#what-pr-a-measured).

The last three rows are **new measurements** (method below). They matter
because the depth descriptors D1–D5 cannot see them: the enriched generator
produces rooms whose *planned paths are dead straight*. The median enriched
episode is a straight-line tracking exercise. That is simultaneously a
domain gap versus every Infinigen scene and a training-signal regression for
a subgoal-tracking policy, and it is the same interior-clearing mechanism
that moved D1 and D4 the wrong way.

Two consequences shape the whole design:

1. **The cheapest, largest lever is a re-range of two existing constants, not
   new machinery.** Both already exist and both already default to the
   vanilla value. That arm must run *before* the surgery, or the surgery is
   tuned against a confounded baseline.
2. **F2 changes status.** It was deferred on "no D5 edge deficit", which
   still holds. But the path statistics show a *topology* deficit that F1/F3
   cannot reach and that enrichment made worse. F2 is re-motivated by a
   different instrument than the one that deferred it.

### Provenance of the numbers

Numbers below are tagged:

- **(probe)** — re-derived in this consult with a reproducible CPU script
  against the real generator, rasterizer and disc kernel. These are safe to
  design against.
- **(consult)** — measured during the multi-agent consult but not
  re-derived here. PR-0/PR-A re-derive them as their first step; no gate
  may depend on one until it has been.

## The design — "sequence and rank"

All changes in
[`mdp/proc_room.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py)
unless stated.

The surgery is deliberately the smallest one that serves all four
consumers. The generator's imperative phase structure, every rejection
test, and every draw site stay as written; two *orderings* that are today
hardcoded become data.

### 1. The seam

`generate_proc_room` gains two arguments, appended last:

```python
placement: PlacementCfg | None = None,   # the lever surface
health_sink: dict | None = None,         # instrument sink; production passes nothing
```

`None`/`None` is the vanilla and NOCAM value. The vanilla
`_GENERATE_PROC_ROOM` params dict
([`strafer_env_cfg.py:1613-1617`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py))
keeps its single `collection_name` key, so the 14 vanilla contract goldens
do not move: `_canon` renders a callable as `<fn module.qualname>`
([`test_composition_contract.py:43-49`](../../../../source/strafer_lab/test_sim/env/test_composition_contract.py)),
so a signature change is invisible to every contract hash. That invisibility
is exactly why the guard net below must pin the signature separately.

### 2. Placement — a pre-expanded slot sequence

Today the furniture and clutter phases hardcode *which slot fills the k-th
placement*:

```python
for f_idx in range(min(n_furn, NUM_FURNITURE)):
    slot = FURNITURE_SLOTS[f_idx]
```

becomes

```python
for slot in furn_seq:            # resolved once per env, before the loop
```

where `furn_seq` is `_VANILLA_FURN_SLOTS[:budget]` when `placement is None`
— a plain slice of a module constant. `FURNITURE_SLOTS[:k]` iterated in
order *is* `FURNITURE_SLOTS[f_idx] for f_idx in range(k)`, so the
equivalence is textual. There is no running accumulator, so there is no
accumulator to hoist out of the loop by mistake; everything inside the
10-attempt body is untouched text.

**Why a sequence is enough for D4:** the clutter sampler never reads the
slot's size — positions come from the room span and the rejection tests
compare XY only — so permuting the clutter sequence relabels an *identical*
geometric sequence. Verified **(probe)**: with the two tall cylinders
promoted to the front of `CLUTTER_SLOTS`, the first divergent RNG draw is
number 1322 of 1338, which is the Phase-6 `torch.randperm` in
`_extract_spawn_points` — i.e. every placement draw is identical and the
stream only re-phases once the layout has changed the reachable-cell count.
The furniture sampler *does* read `OBJECT_SIZES[slot, :2]`, so a furniture
permutation diverges early (draw 909) — that is fine, since it is
enriched-only, but it is why the two categories cannot be assumed
symmetric.

### 3. Park order — a group rank

The retry ladder's two-tier walk
(`reversed(CLUTTER_SLOTS)`, then `reversed(FURNITURE_SLOTS)` if nothing was
parked, [`proc_room.py:956-971`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py))
is a hand-encoding of one total order over slot *groups*. It becomes:

```python
park_rank = _VANILLA_PARK_RANK if (placement is None or placement.park_rank is None) else placement.park_rank
max_retries = len(park_rank)
if failed.any():                                  # keep the existing fast-out
    for _ in range(max_retries):
        if not failed.any():
            break
        for fi in torch.where(failed)[0]:
            for group in park_rank:
                if active_mask[fi, list(group)].any():
                    active_mask[fi, list(group)] = False       # atomic
                    poses[fi, list(group), :3] = PARK_POS.to(device)
                    break
        ...                                        # rebuild + re-check unchanged
```

with `_VANILLA_PARK_RANK` the 24 singleton groups in today's order.
"Walk the rank, park the first group with an active member" reproduces the
two-tier walk exactly, because the furniture entries are unreachable until
every clutter entry is inactive — precisely the old `if not parked`
condition. Groups exist for H1: a compound must be parked as a unit.

The ladder consumes **no randomness** (verified **(probe)**: the module's
only draw sites are lines 508, 584-585, 698-699, 729-731, 826-845, 879-921),
so the park-order half of this surgery is provably RNG-free. Its obligation
reduces to equality on `active_mask` and `poses`, and equal `active_mask`
implies equal occupancy, free space, reachability, and therefore an
identical `randperm(K)`.

### 4. Vanilla-preservation argument

The constraint is real and the mechanism is understood:

- Every scalar draw is issued inside `for b_idx in range(B)`, so the stream
  is **positionally coupled across envs** and onward into
  `reset_robot_proc_room`'s later draws. A strategy-major refactor — the
  "clean" restructure any reviewer reaches for first — is a total stream
  break. Env-outer iteration is a law, not a style choice.
- Enrichment draws must stay **guarded before the first draw**, the pattern
  `clutter_wall_bias_prob > 0.0` already establishes
  ([`proc_room.py:877-879`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py)).
- Draw counts are **data-dependent**: the 10-attempt rejection loops consume
  draws per *rejected* attempt, and `randperm`'s size is the reachable-cell
  count. Any layout change re-phases everything downstream in that reset.
  "Byte-identical" therefore means *identical layout implies identical
  stream* — never *different layout, same stream*.

Two properties were confirmed **(probe)** on the existing CPU stub harness:
passing every enrichment argument explicitly at its default produces **zero**
divergent draws versus omitting them; and the shipped F3 height map produces
zero divergent draws *and* leaves the XY of every slot unchanged — the
existing proof that a pure-appearance lever is achievable.

**The load-bearing gap, stated as a finding.** The existing
`test_default_path_byte_identical_to_explicit_defaults`
([`test_proc_room_enrichment.py:174-204`](../../../../source/strafer_lab/tests/navigation/test_proc_room_enrichment.py))
is a *within-version* self-consistency check: both legs run post-change
code and move together, so it cannot detect a change that perturbs the
default path uniformly. Freezing new goldens in the same PR as the rewrite
reproduces that defect one level up. The guard net must therefore land in
its own PR **against unmodified code** — see PR-0. This is the single
mechanism that actually enforces the non-negotiable.

### 5. NOCAM traversal and the proof obligation

NOCAM's gym IDs pass exactly `{"collection_name"}` and pin difficulty 7/7,
so `placement is None`, `health_sink is None`. Enrichment is *structurally*
unreachable for them, not merely unset: the gate is
`enrich_depth and obs_profile() in ("depth", "full")`
([`composed_env_cfg.py:372`, `:499`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py))
and a camera-free stack's profile is `nocam`.

| stage | NOCAM state |
|---|---|
| placement-cfg validator | not called |
| slot-sequence resolver | plain slice of a module constant |
| compound posing / internal walls | not entered |
| solvability predicate | area floor only, byte-identical |
| park rank | `_VANILLA_PARK_RANK`, length 24 |
| health sink | `is None`, one test per reset |

The frozen manager hashes cannot see any of this — they cover manager cfgs
and sim/scene scalars, not the generator's internals. The obligation is
therefore a set of CPU, Kit-free tests on the existing stub harness:

- **N-1 output golden** — a seed × difficulty × batch-size sweep (batch size
  matters: the stream is positionally coupled across envs, so a
  batch-dependent defect is invisible at B=1), hashing
  `poses ‖ active_mask ‖ free_space ‖ spawn_pts ‖ spawn_count` plus a draw
  taken immediately after the call. *As shipped:* `reachable` is not stored
  on the env, so the trailing draw stands in for it — and does the job
  better, since it pins total RNG *consumption* directly rather than the
  quantity `randperm(K)` derives from. Every seed × difficulty runs at
  B ∈ {1, 8}; B = 64 is a spot check over the open-field / sparse / dense
  regimes, because a dense 64-env case costs ~2.3 s and adds scale rather
  than new branches.
- **N-2 draw witness** — wrap `torch.rand`/`randint`/`randperm` and freeze
  the ordered `(op, shape)` sequence plus per-operation totals. *As shipped:*
  attribution is by draw **index**, not by phase label — phase labels would
  have to be keyed on source lines, and would not survive the refactor they
  exist to guard; a failure reports the first divergent index with its
  neighbourhood, which localizes the culprit just as well. Scope honestly:
  this catches operation, shape and conditionality changes — including the
  batched-versus-scalar hazard — and nothing value-dependent.
- **N-3 signature and defaults pin** — the contract hash is blind to the
  signature, so pin `inspect.signature(generate_proc_room)` against a frozen
  dict.
- **N-4 vanilla consumer table** — every vanilla RL cfg class **and**
  `StraferNavCfg_BridgeAutonomy_ProcRoom`: generator params are exactly
  `{"collection_name"}`, difficulty 7/7, palette equals the frozen palette
  golden, no `ceiling` attribute. The existing six-name palette test is
  narrower than the consumer set.
- **N-5 palette insertion-order golden** — `_palette_sig` sorts by name
  ([`test_composition_contract.py:187`](../../../../source/strafer_lab/test_sim/env/test_composition_contract.py))
  while insertion order *is* the slot index every range and `OBJECT_SIZES`
  row depends on, protected today only by a comment.
- **N-6 `OBJECT_SIZES` immutability** — an autouse fixture hashing the
  module tensor around every generator call. `sizes = OBJECT_SIZES.to(device)`
  returns the module tensor itself on CPU, and the NOCAM-visible proximity
  reward imports it.
- **N-7 fast-out preserved** — with no failing env, the occupancy grid is
  built exactly once.

**Stated ceiling:** nothing in CI constructs a Kit env. The 256-env NOCAM
construct-and-step smoke stays an operator gate, triggered by global palette
growth or a raised per-env prim count. This design triggers neither —
compounds reuse existing slots, F2 reserves from the existing 20 wall slots,
and D4 is a permutation.

### 6. Solvability

**Termination.** `max_retries = len(park_rank)` (24 for the vanilla rank —
today's `NUM_CLUTTER + NUM_FURNITURE` literal, and it grows automatically
when F2 appends a tier). Each pass, each still-failing env parks the first
group with an active member and deactivates all of it. Groups are pairwise
disjoint and nothing re-activates, so the index of "first group with an
active member" strictly increases; after at most `len(rank)` passes every
group is inactive. The terminal geometry is the bare perimeter box, which is
the state today's argument already relies on.

**Protection is a rank, never an exemption.** A protected class moves later
in the rank and stays parkable, so the terminal state is unchanged and the
existing argument carries verbatim. Hard protection would make
exhaustion-with-failure reachable for the first time, against code that
falls through silently. Protection is also **bounded** — a protected class
goes after its *own category*, not after everything, so protecting a 1.8 m
column cannot cause the ladder to strip a 2.0 m shelf instead.

**The invariant that is missing today.** `MIN_REACHABLE_CELLS = 100` is an
*area* floor, so a partition that cleanly bisects a room is invisible to it.
Measured **(probe)**, a sealed partition at spans 4.0 / 5.5 / 7.5 m gives
reachable-over-free coverage **0.501 / 0.488 / 0.492** with 450 / 988 / 2078
reachable cells — all far above 100, so `failed` stays False and the episode
trains in a silent half-room with robot, goal and path endpoints all in the
same component. Healthy rooms measure 1.000. The separation is enormous, so
an enriched-gated coverage floor (`reachable < floor × free_space.sum()`,
around 0.85) is nowhere near knife-edge. It must be enriched-gated because
it changes `failed`, hence reachability, hence the `randperm` stream — and
its floor must be baselined against the *vanilla* coverage distribution
first, since ordinary clutter can in principle enclose a pocket.

**BFS seed.** Keep the centre seed and keep writing it unconditionally.
`_gpu_bfs` marks the seed cell reachable whether or not it is free
([`proc_room.py:438`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py)),
which is what guarantees K ≥ 1 and keeps the spawn pool non-degenerate.
Seed protection belongs at *placement* time: every mid-room placement class
rejects candidates whose inflated footprint meets a disc about env-local
(0,0). The radius must cover the seed cell **and its 3×3 ring**, because BFS
expands with a 3×3 max-pool — so at least `INFLATION_CELLS × GRID_RES +
2 × GRID_RES = 0.5 m`.

**A latent defect this must not activate.** If a spawn pool ever degenerated
to a single point, the goal sampler would pick that same point, `plan_path`
would reject it as coincident, the straight-segment fallback would emit a
two-point path, and the dwell-gated `path_complete` reward would pay in
full — a success with no navigation. The chain was traced end to end and is
**not reachable today** (the unconditional seed write keeps K ≥ 1 and the
ladder's terminal state is the open box), so this is a guard obligation on
F2 and on any future seed change, not a live bug. Gating `path_complete` on
`path_fallback == 0.0` closes it cheaply and independently.

## Per-consumer parameterization

### D4 — mid-room tall-column frequency

*Mechanism.* The two tall cylinders sit at clutter indices 14–15, so they
are reached only when `n_clut ≥ 15` — level 7 alone, about a quarter of
enriched episodes — and `reversed(CLUTTER_SLOTS)` parks them **first** when
a room is unsolvable. Measured **(probe)** over 256 rooms at the enriched
`U[4,7]` mixture: a mid-room tall column is present in **19.9%** of rooms,
and of the rooms where the level made one a candidate, **16** lost it to
placement rejection or the ladder. Promoting the two slots to the front of
the clutter sequence — which, because the ladder scans the sequence
reversed, simultaneously makes them parked *last* — moves presence to
**99.6%** and candidate-but-absent to 1.

*Knob.* A per-episode promotion **probability**, not a binary reorder:
99.6% presence is an overshoot in the opposite direction, and the project's
widen-not-shift discipline applies here as it did to difficulty. Columns get
their own additive placement phase with an explicit count and a radial
placement law bounded away from the BFS seed, so the A/B does not
*displace* two ordinary clutter items and confound itself.

*Gate.* D4 `high_row_near_frac` from the existing descriptor instrument —
GPU window required. CPU witnesses (free): per-category placement and park
counts from the health sink.

*Stated ceiling, up front.* D4 is dominated by wall proximity, not by
columns: a ray attribution **(consult)** puts 73.4% of D4-near cells on
walls, 18.4% on shelves, 4.5% on cabinets and **2.3% on columns**, which is
consistent with the camera geometry — at 0.35 m height and 56.4° vertical
FOV, the upper 18 rows at 2 m see surfaces from 0.57 m to 1.39 m tall, a
band a 2.7 m enclosing wall fills completely whenever the robot is within
2 m of it. F3 raised three tall categories at once and bought +1.8 points.
This knob is a shaper, not a standalone D4 plan, and it should not be sold
as one.

### D1 — near-band rebalance

*Attribution* **(consult)**: spawn erosion −0.11 m, clutter wall bias
−0.10 m, difficulty un-pin −0.04 m, span nudge −0.02 m, closing 0.27 of the
0.28 m overshoot.

*Arm A — zero code.* Re-range `_ENRICH_ROBOT_SPAWN_INFLATION` 2 → 1 or 0 and
`_ENRICH_CLUTTER_WALL_BIAS` 0.5 → 0.3. Both arguments exist, both default to
the vanilla behavior, and this is simultaneously the largest D1 lever and
the largest single D4 cost. The spawn erosion's original justification was
retracted by the brief's own calibration correction: it was installed to
close a standable-clearance gap that the rendered descriptors then showed
does not exist (pooled Infinigen 1.19 m versus ProcRoom vanilla 1.17 m).

**Fairness floor (ruling F-5).** Reducing the erosion takes the D1 side of a
conflict with the parked reactive-avoidance brief's M4, but the half of M4's
concern that survives is a floor, not a margin: the robot must never spawn
*in contact range*, so a pivot-radius clearance is retained even at erosion
0. Concretely, erosion 0 leaves the shared pool's 0.3 m robot-radius
inflation, which already exceeds the chassis circumscribed radius — the floor
holds by construction, and PR-A must assert it rather than assume it (the
spawn-to-nearest-object CPU statistic below is the natural place).

*Arm B — geometry side, only if A under-delivers.* The same radial
placement law applied to two or three non-tall clutter items, keeping the
fairness floor above.

*Gate.* D1 median, plus a **floor-masked** D1: at this camera geometry the
flat floor alone supplies a large share of sub-2 m pixels, so the pooled
median is partly a camera constant rather than a placement statistic. CPU
pre-gates (free): spawn-to-nearest-object median and spawn-to-nearest-wall
p10.

*Never* re-tune the 0.5 / 0.4 / 0.3 m rejection thresholds — those loops
draw per rejected attempt, so a threshold change is a stream change with no
descriptor payoff.

### H1 — primitive compounds

*Route.* Existing slots, existing footprints, pose-only co-location:
members at `anchor + R(anchor_yaw) · offset`, no new palette slot, no
per-env `OBJECT_SIZES`. `OBJECT_SIZES` is module-level and read by the
NOCAM-visible proximity reward, so per-env footprints and new slots are both
excluded. Every slot is already kinematic, so rigidity is free.

Occupancy *wants* the multi-slot form: each member rasterizes as its own
rotated AABB and unions, which is strictly more faithful than a single
anchor AABB. The opposite formulation — one anchor whose AABB under-covers
the rendered geometry — is forbidden, because it is precisely how the robot
could drive through visible geometry while BFS calls the room solvable.

*Priced costs.* The proximity penalty sums over every active slot in range,
so a K-member compound is charged K times; members consume K of the 8/16
budget; members must be appended to the placed-XY lists or later items will
be accepted inside them; and members must be floor-contacting, with height
variation routed through the proven-neutral `tall_object_heights` map rather
than a member z-offset.

*Gate — does not exist yet, so H1 is instrument-first.* Footprint-area,
top-z and wall-alignment distributions compared as 1-D Wasserstein distances
against the Infinigen corpus metadata, with "reduce by X%" targets rather
than "mimic" — the achievable set is the discrete union-lattice of 44 fixed
AABBs, not a continuous family. H1 is the only consumer whose gap is
*asserted* rather than measured, so under "instruments before knobs" it does
not get a knob until the instrument exists. Wall-alignment in particular has
**no knob at all** today (room-mode furniture yaw is one of four hardcoded
constants), which may be a reason to drop alignment from H1's stated goal.

### F2 — internal walls (unblocked by ruling F-3; ships in its own PR)

*Precondition already in tree.* `max_internal_walls` is computed and read
but never consumed; the level table already supplies 1/1/2 internal walls at
levels 5/6/7. F2 is a fill-in, not a new axis.

*Structure.* Statically reserve internal-wall slots from the existing 20
(the four short slots are the natural reservation) so they are never visible
to the perimeter packer — which pops from shared budgets, meaning a runtime
"is this segment internal" mask cannot be made static. Perimeter slots stay
out of the park rank entirely: parking one opens a floor-to-ceiling
far-clamp slit while the exterior clip still treats the room as sealed,
which is the regression the span cap exists to prevent. Internal walls get
their own height, asserted below the ceiling slab's underside, or they
pierce it in exactly the band D3 and D4 read.

*Aperture width — measured, and it corrects a consult error.* The corridor
surviving a gap of `n` free columns is `n − 2 × INFLATION_CELLS` cells.
Measured **(probe)** against the real rasterizer and disc kernel, swept over
five sub-cell phases:

| gap (m) | free columns | corridor cells | through-connected |
|---|---|---|---|
| 0.60 | 6 | 0 | 0/5 |
| **0.70** | 7 | 1 | **5/5** |
| 0.80 | 8 | 2 | 5/5 |
| 0.90 | 9 | 3 | 5/5 |

The geometric floor is **0.70 m**, which is exactly what the parent brief
already states (`2 × INFLATION_CELLS + 1` cells). A consult pass derived a
1.26 m floor from an off-by-one in this formula and concluded the admissible
window was empty; that conclusion does not survive re-measurement.

The binding constraint is not geometric, though. The subgoal off-path
tolerance is 0.30 m, so a gap narrower than about **1.16 m** tells the policy
that 0.30 m of lateral error is acceptable inside an aperture that physically
does not permit it. **Ruling F-3 sets the shipped minimum at 1.2 m** — margin
above that collision point, and comfortably clear of the 0.70 m raster floor,
so the wall's own sub-cell phase is no longer a concern. The aperture-aware
off-path tolerance that would license narrower gaps is a shared-reward change
and belongs to v3.

A 1.2 m minimum means internal walls do **not** produce the tight-doorway
threading the Infinigen corpus shows (min path clearance 0.41 m); what they
produce is *topology* — the detours and turns PS1 measures. PS2's threshold
must therefore be re-derived so the statistic still counts ordinary
furniture-gap threading rather than only sub-1.2 m doorways, or it will read
flat across the very lever it is gating.

## The two new path statistics

Home: a new numpy-only, Kit-free tool beside
[`scene_connectivity.py`](../../../../source/strafer_lab/strafer_lab/tools/scene_connectivity.py).
Both statistics run on a robot-radius-inflated free grid plus the raw
occupancy grid, through the **unmodified** shared A*
([`path_planner/planner.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/path_planner/planner.py),
whose entire import list is `heapq`, `math`, `numpy`). Verified CPU-only on
both corpora — no Kit, no GPU window. The EDT must be implemented in numpy:
scipy is **not** available in the lab environment (verified) and is not a
declared dependency; `scene_connectivity._disc_dilate` is the existing
numpy-only precedent.

**PS1 — corridor curvature.** For the planner's polyline `p_0..p_{N-1}`
with segments `d_i` and wrapped exterior angles `θ_i`:

- `turn_density = Σ|θ_i| / Σ‖d_i‖` (rad·m⁻¹) — primary
- `tortuosity = Σ‖d_i‖ / ‖p_{N-1} − p_0‖` — secondary
- `frac_paths_bending` = fraction with tortuosity > 1.05

Gate on the **fraction and an upper percentile, not the median**: the
enriched median is exactly 0.000, pinned at its floor, because the planner
line-of-sight-straightens unconditionally.

**PS2 — aperture threading.** With `C(x)` the Euclidean distance from a path
waypoint to the nearest *uninflated* obstacle cell:

- `frac_below(τ)` — fraction of arc length with `C < τ` — primary
- `min C` along the path — secondary

`C` is a *path-centreline clearance*, not a corridor width: A* hugs the
inflation boundary, so `C` sits near each source's inflation radius by
construction. It is a comparative statistic between sources measured
identically, and it is only comparable after adding each source's own
inflation radius back: **0.30 m** for ProcRoom
(`INFLATION_CELLS × GRID_RES`) versus **0.18 m** for the Infinigen adapter
(the inscribed half-width, different by design). Fix the threshold once in
that corrected half-width, from the Infinigen reference distribution, before
any lever is tuned.

Ruling F-3 constrains that choice: with internal walls held at ≥ 1.2 m the
statistic must still register *furniture-gap* threading, so the threshold is
derived from the Infinigen reference distribution's body rather than set at
its doorway tail. A threshold that only fires below 1.2 m would read flat
across the F2 lever and be worthless as its gate — verify sensitivity on the
PR-A baseline before PR-3b tunes anything.

**Reference values (probe)**, ProcRoom at 64 envs with 6 endpoint pairs each
(n = 151 vanilla / 206 enriched) and Infinigen singleroom at n = 60:

| | turn density (rad/m) | tortuosity | bending paths | min C (m) | arc below 0.6 m |
|---|---|---|---|---|---|
| ProcRoom vanilla (L7) | 0.241 | 1.035 | 43.7% | 0.63 | 71.7% |
| ProcRoom enriched | **0.000** | **1.000** | 16.5% | 0.72 | 21.3% |
| Infinigen singleroom | 0.242 | 1.287 | 76.7% | 0.41 | 72.1% |

> Superseded by PR-A: the singleroom row is an *unsealed* measurement. See
> [What PR-A measured](#what-pr-a-measured) for the shipped tool's baseline.

Whole-house Infinigen seeds measure tortuosity 1.18–2.10 but with 13–31 m
arcs against ProcRoom's ~3 m, so **the two room-scale scenes are the honest
comparator** and the whole-house seeds are context only.

Three comparability corrections are mandatory in the shipped tool, and the
prototype above does not yet apply all of them: **resolution** (0.1 m
ProcRoom versus 0.05 m Infinigen — clearance is quantization-locked at one
cell on both sides), **inflation radius** (above), and **arc length**
(report against a matched straight-line-distance band, or the statistic
partly measures how far apart the endpoints were). A fourth caveat is
semantic and cannot be corrected, only stated: the ProcRoom grid rasterizes
every object's full XY AABB, while the Infinigen grid is a horizontal slice
at robot height, so low objects block in one and not the other. That bias
makes ProcRoom look *more* cluttered, so the measured gap is conservative.

## Implementation split

Two PRs precede the expected PR-1. Both are justified by findings, not
caution.

**PR-0 — guard net only, zero production diff.** Lands N-1…N-7 against
unmodified code. Without it PR-1 is unverifiable, because the existing
dual-run test provably cannot detect a uniform default-path change and
freezing goldens alongside the rewrite reproduces that defect. *Blocking for
everything below.*

Per ruling F-2 the N-1 goldens are captured from the **pre-enrichment tree**
and then a **reconciliation assert** verifies the current tip reproduces them
over the full sweep. The measurements say the 1-ulp door-width deviation is
absorbed everywhere, so equality is the expectation — the assert converts
that expectation into a fact. If any seed diverges, stop and surface it: that
reopens the freeze decision and probably motivates making the enriched
door-width expression ulp-exact.

**PR-A — zero-code lever arm plus instrument build-out.** Re-range the two
enrichment constants (Arm A above); add seeding and per-room confidence
intervals to the descriptor capture (its CI must derive from room count,
not pixel count — the same pooled-degrees-of-freedom error the depth-noise
track already had to correct); add per-env poses to the capture record so
D4 attribution is an exact ray-cast rather than a model; add the
floor-masked D1; land the path-statistics tool and baseline both corpora.
Gate: a published noise floor for D1 and D4, plus the re-ranged arm's
numbers. **One GPU window.** *Why before the surgery:* this is the largest
measured lever for both instrumented consumers and needs no new machinery.
If it closes D1 and recovers D4 materially, the remaining gap may not
justify the surgery at all.

**PR-1 — the surgery, plus the D4 frequency lever.** Slot-sequence resolver,
group-atomic park rank, `PlacementCfg`, validators, health sink; then the
column phase with bounded park protection as **one** arm (raising presence
without protecting park order raises the park rate too). Gate: every PR-0
golden green **with zero edits to the frozen literals** — a moved golden is
an automatic reject — plus D4 and D1 reported against PR-A's noise floor.
**One GPU window.** Compounds, internal walls and the coverage invariant are
not in this PR.

Two operator-run merge gates ride along. Per ruling F-1, a one-off **CUDA
pose-hash** at 64 envs before and after the surgery — a few minutes in any
Kit window — is what extends CPU certification onto the device; there is no
standing GPU gate. Per ruling F-6(ii), a **256-env NOCAM construct-and-step
smoke carrying a reset-time measurement** answers the reset-path cost that no
benchmark in the repo has ever measured. Note the asymmetry: this design does
not otherwise *trigger* that smoke (no palette growth, no added per-env
prims), so PR-1 runs one specifically to host the measurement and to
re-confirm NOCAM at scale after the ladder is rewritten.

**PR-2 — H1, instrument first.** Build and baseline the footprint/top-z/
alignment distributions; then the compound mechanism and invariants; the
compound catalogue ships **only if** the instrument shows a measured gap.
Gate includes "D4 must not regress" — consolidating three tall wall-flush
anchors into one compound would trade away a larger D4 contributor than
columns are. Per ruling F-6(i), PR-2's validation also carries a
**measured-in-Kit** check of PhysX contact behavior at a compound join: two
overlapping kinematic bodies is a configuration nothing in the repo
exercises, and no amount of reading settles it.

**PR-3a — the solvability work, which ships regardless of walls.** The
coverage invariant, the seed-protection predicate, and the path statistics
promoted to durable gates. Worth landing whether or not F2 ever ships: the
coverage invariant closes a silent bisection channel that nothing else can
see.

**PR-3b — internal walls. Unblocked** by ruling F-3, under the option-(a)
constraint: apertures **≥ 1.2 m**, and PS2 re-thresholded so it still counts
ordinary furniture-gap threading. Still gated on the residual wall-budget
composition (achievable contiguous span, not total length) and the
4-connectivity-safe rejection inside the wall attempt loop. The narrower
aperture regime remains available only via an aperture-aware off-path
tolerance, which is a shared-reward change filed to v3.

*Independent one-liner:* gate `path_complete` on `path_fallback == 0.0`.

Ordering: PR-0 → {PR-A, PR-1}; PR-A → PR-1; PR-1 → {PR-2, PR-3a}; PR-3a →
PR-3b. PR-2 and PR-3a are independent. Total GPU windows: two, possibly
three, plus the operator-run Kit gates on PR-1 and PR-2.

## Structural extensions — each needs its own consult gate

Sanctioned here (no gate): new generator kwargs; new keys in the *enriched*
event params, which flip only the 8 enriched goldens — their own recorded
provenance says no checkpoint depends on them, and ruling F-6(iii) confirms
none has since; the enriched-gated coverage invariant, terminal ladder tier
and health sink. No new composition axis, no
new `SceneSourceCfg` field, no new gym IDs — this design rides the existing
`enrich_depth` seam, and a placement-only lever adds no variant.

Gated (do **not** proceed without a consult):

1. **Growing the palette.** Breaks the no-new-slot golden, and NOCAM shares
   the palette object; grows the slot-range constants in lockstep; grows
   PhysX actors against buffers explicitly sized for 44 primitives at 256
   envs; and would have to re-run the 256-env smoke that PR-1 stands up for
   ruling F-6(ii), against a grown palette. This design is built to avoid it. If H1's footprint gap proves unclosable on the fixed AABB
   lattice, this is the escalation — with numbers, not after a failed gate.
2. **Splitting `_proc_room_active_mask` into occupancy and reward masks.**
   The clean fix for the K-fold compound penalty, but it touches a tensor
   the NOCAM-visible proximity reward reads, and both NOCAM reward sets are
   wired to it.
3. **Exposing the planner's pre-shortcut cell path.** A shared-planner touch
   across five consumers. Recommendation: don't — adopt the stronger
   criterion that internal walls must create geometry the shortcut *cannot*
   straighten.
4. **Making the solvability BFS 4-connected** to match the planner's
   no-corner-cut rule. It would change reachability on the shared path, so
   it is unavailable under the byte-identity constraint. Measure the
   BFS-solvable-but-unplannable rate instead and hold it flat.
5. **Any change to the proximity reward**, including an aperture-aware form.
   Shared with NOCAM.

## What PR-A measured

Everything below is re-derived, so it supersedes the `(consult)` tags above.
The descriptor legs are 512 rooms each (64 envs × 8 resets), the CPU statistics
are on the stub harness, and the path statistics are the shipped tool.

### Arm A as shipped

`_ENRICH_CLUTTER_WALL_BIAS` 0.5 → 0.3 and `_ENRICH_ROBOT_SPAWN_INFLATION` 2 →
**1**. The design sanctioned "1 or 0"; the fairness floor picks 1, and it is a
measurement, not a preference.

**Ruling F-5's stated mechanism is wrong.** "Erosion 0 leaves the shared pool's
0.3 m robot-radius inflation, which already exceeds the chassis circumscribed
radius — the floor holds by construction" does not survive measurement: the
0.3 m disc is rasterized on a 0.1 m grid and a spawn point is a cell *centre*,
so the guarantee is about a cell short of its nominal radius. Over 6 seeds ×
64 envs (~75 k pool candidates), the fraction of candidates inside the 0.28 m
chassis circumscribing radius:

| arm | spawn→object median | p10 | min | inside 0.28 m | wall p10 |
|---|---|---|---|---|---|
| vanilla L7 | 0.571 | 0.389 | 0.166 | **0.190%** | 0.585 |
| enriched, erosion 2 (prior) | 0.906 | 0.637 | 0.272 | 0.001% | 0.806 |
| enriched, erosion 1 + bias 0.3 (**shipped**) | 0.786 | 0.525 | 0.384 | **0%** | 0.664 |
| enriched, erosion 0 + bias 0.3 | 0.710 | 0.417 | 0.163 | **0.108%** | 0.561 |

Erosion 1 is the largest reduction that keeps the floor, so that is what
ships. Note the violation is *pre-existing in vanilla* (0.190%), so erosion 0
would not have introduced a new failure class — it would have returned the
enriched arm to the vanilla floor, which is exactly what F-5 declined. The
floor is now a test (`test_proc_room_spawn_clearance.py`), including a
negative test that fails if the erosion is removed.

### Published noise floor

Two independent seeds, identical configuration, 512 rooms each:

| descriptor | seed A | seed B | replicate Δ | 95% room-level half-width |
|---|---|---|---|---|
| D4 high-row near (%) | 43.85 | 43.35 | 0.50 | **±3.1** |
| D1 median (m) | 1.745 | 1.733 | 0.012 | **±0.067** |

The room-level bootstrap is not understating: the observed spread of the
per-reset batch means (sd 4.4 points over 8 batches of 64) matches what
independent rooms predict (4.5), so rooms behave as independent draws and the
interval is honest. This replaces the consult's ±8.3-point unpaired figure and
is 2.7× tighter.

### Arm A's descriptor table

Arm A was measured twice, under both capture protocols, because the protocol
turned out to matter (below). Bootstrap over rooms; `*` = interval excludes
zero.

**Paired legs** — shared seed, matched room for room, 512 pairs. This is the
protocol ruling F-4 mandates:

| descriptor | prior | Arm A | paired Δ |
|---|---|---|---|
| D1 ≤1 m (%) | 35.16 | 36.52 | +1.36 [−1.20, +3.88] |
| D1 median, room-level (m) | 1.745 | 1.753 | +0.008 [−0.084, +0.100] |
| D1 median, floor-masked (m) | 2.218 | 2.235 | +0.016 [−0.113, +0.148] |
| floor pixels (%) | 31.50 | 30.60 | −0.90 [−2.04, +0.23] |
| D3 top-11 pinned (%) | 5.84 | 6.16 | +0.32 [−0.88, +1.51] |
| D4 high-row near (%) | 43.85 | 42.73 | −1.12 [−5.60, +3.20] |
| D5 edge density (%) | 1.65 | 1.74 | +0.09 [−0.05, +0.23] |

**Independent-sample legs** — the unseeded protocol, 512 rooms each, no shared
seed, so this is an unpaired comparison and its intervals are wider by
construction:

| descriptor | prior | Arm A | Δ |
|---|---|---|---|
| D1 ≤1 m (%) | 32.05 | 35.68 | **+3.63 [+1.18, +6.01]** `*` |
| D1 pixel-pooled median (m) | 1.525 | 1.412 | −0.113 |
| D1 median, room-level (m) | 1.823 | 1.753 | −0.070 [−0.161, +0.021] |
| floor pixels (%) | 32.74 | 31.23 | **−1.51 [−2.57, −0.43]** `*` |
| D3 top-11 pinned (%) | 5.95 | 6.57 | +0.62 [−0.65, +1.84] |
| D4 high-row near (%) | 38.83 | 41.24 | +2.41 [−1.98, +6.69] |
| D5 edge density (%) | 1.68 | 1.75 | +0.07 [−0.07, +0.20] |

Read honestly across both legs: **Arm A's effect on D1 and D4 is not resolved
at 512 rooms.** Two of the fourteen interval readings exclude zero, both in
the unpaired leg — the D1 near-band fraction (+3.63, toward the 44.2% target,
about 30% of the remaining gap) and the floor-pixel fraction (−1.51, i.e.
measurably more non-floor content in frame). The two legs agree in sign on the
D1 near band and **disagree in sign on D4** (−1.12 versus +2.41), with both
intervals straddling zero. D4 is unmoved by this lever.

F-4's D4 gate is therefore **not met**: the paired interval on the improvement
does not exclude zero, whichever leg is read. The surgery stays motivated on
D4 and on the rest of the D1 gap. Note separately that the *baseline's* D4
point estimate lands at 43.85 (paired protocol) or 38.83 (unpaired) against a
40.6 band floor — the floor sits inside the protocol spread, which is the
instrument problem recorded below, not a lever result.

Pairing is measured rather than assumed. At a shared seed, prior vs Arm A:
difficulty identical for **100%** of rooms, geometry byte-identical for
**10.2%**. Across seeds: 23.6% (chance, for four difficulty levels) and 0%.
Exact room-for-room pairing is available only for a knob that touches nothing
before Phase 6 — the erosion half pairs exactly, the clutter-bias half
re-phases the per-env stream and unpairs about 90% of the batch. Arm A
contains both, so 10.2% is its ceiling and the interval above is the honest
one.

### D4 attribution, by ray cast rather than by model

Casting the true pixel ray from the true camera pose against each room's own
boxes (24 rooms, 281 k D4-near pixels):

| surface | ray cast | consult model |
|---|---|---|
| wall | **80.7%** | 73.4% |
| shelf | 15.4% | 18.4% |
| tall column | **1.8%** | 2.3% |
| cabinet | 1.3% | 4.5% |

The measurement confirms the model's shape and makes it *more* extreme. The
design's "stated ceiling" for the column knob stands and hardens: at 1.8% of
D4-near cells, the mid-room column frequency lever is a shaper, not a plan.

### Path statistics — the comparator was wrong

The tool ships with the three corrections applied. Baselining both corpora
exposed a defect in what the design's PS numbers were compared against.

Measured the way ProcRoom is measured — the robot confined to the room
footprints its own spawn derivation gives it — the honest room-scale
comparator is **not** curved (arc band 1.5–4.0 m, 95% group-resampled):

| leg | paths turning (%) | paths bending (%) | turn density p90 |
|---|---|---|---|
| Infinigen singleroom, **sealed** | **3.7** | **0.3** | 0.000 |
| Infinigen singleroom, unsealed | 61.2 | 51.2 | 0.534 |
| Infinigen tworoom, sealed | 51.3 | 26.7 | 0.514 |
| Infinigen whole-house (5 seeds), sealed | 16.0–37.8 | 12.0–35.1 | 0.307–0.426 |
| ProcRoom vanilla L7 | 62.7 [58.7, 66.6] | 33.4 [30.0, 37.2] | 0.713 [0.668, 0.773] |
| ProcRoom enriched, prior | 37.3 [33.2, 41.5] | 19.0 [16.0, 22.1] | 0.487 [0.421, 0.548] |
| ProcRoom enriched, Arm A | 40.8 [36.4, 45.1] | 20.0 [16.9, 23.0] | 0.533 [0.468, 0.602] |

The design's reference row (singleroom, tortuosity 1.287, 76.7% bending)
reproduces only when the path is allowed to leave the building and round its
exterior — which is not a mission the robot runs and not the region the bridge
spawns it in. Sealed to its one room, that scene is 97% line-of-sight.

Two conclusions follow, and they point in opposite directions:

1. **The enrichment's straightening is real and reproduces** — vanilla 62.7%
   turning versus 37.3% for the enriched generator, intervals well separated.
2. **The domain gap it was compared against is not.** Against the room-scale
   corpus measured identically, the enriched generator (40.8% turning, 20.0%
   bending) sits *inside* the scanned range, and vanilla sits at or above its
   top. Curvature in the corpus is a **multi-room** property: a single
   furnished room does not bend paths at this robot radius, in either corpus.

Both ProcRoom biases run the same way (0.30 m inflation versus 0.20 m, and
full-AABB rasterization versus a slice at robot height), so ProcRoom is
flattered on these statistics and the residual gap is smaller still. That
strengthens F2's topology motivation on its own terms and removes the
"enrichment opened a path-topology domain gap" framing that partly motivated
it. **Re-ranking on that is the coordinator's call, not this PR's.**

Arm A's effect on path statistics comes entirely from the clutter-bias half.
The spawn-erosion knob **cannot** move them by construction: it writes only
the robot spawn pool, and the planner's endpoints come from the shared pool.

### PS2's threshold, derived from the body

Pooled over the seven sealed scenes, arc-weighted excess clearance:
q10 = 0.025, q25 = 0.105, **q50 = 0.402**, q75 = 0.880 m. The threshold is the
median — **τ = 0.402 m on the inflation-free scale**, i.e. a raw 0.702 m for
ProcRoom and 0.602 m for the scanned adapter. The sensitivity sweep beside it
shows the statistic is not flat at that choice and does compress at a
tail-derived one:

| leg | τ=0.025 (tail) | τ=0.402 (**body**) |
|---|---|---|
| Infinigen singleroom sealed | 1.2% | 22.5% |
| Infinigen tworoom sealed | 11.0% | 57.9% |
| ProcRoom vanilla | 16.2% | 77.1% |
| ProcRoom enriched prior | 8.2% | 50.3% |
| ProcRoom enriched Arm A | 9.3% | 54.6% |

### Instrument facts the next PR should not re-learn

- **The descriptor estimand matters more than most levers.** The recorded
  table's D1 median is pooled over every pixel; the mean of per-room medians
  is a different quantity and reads ~0.3 m higher. A room-level interval
  requires a room-level estimand, so both are now reported side by side and
  the pooled one is labelled as the legacy reconciliation number.
- **The capture protocol moves D4 by more than the gate's resolution.** The
  same configuration measures D4 = 38.8/36.4% when the capture lets the RNG
  stream run across resets and 43.9/43.4% when it re-seeds per reset (which is
  what pairing requires). That is a ~5–7 point protocol effect against a
  4.2-point gap to the band floor. Sequential-seed correlation and reset count
  were both ruled out as causes; the mechanism is not established. **Any D4
  claim must state its protocol.** The unseeded protocol reproduces the
  recorded table exactly (vanilla D1 pooled 1.189 vs 1.17, D1 ≤1 m 42.3 vs
  42.7, D2 top 4.62 vs 4.56, D3 60.0 vs 58.9, D4 42.5 vs 43.7, D5 2.41 vs
  2.35; enriched D1 pooled 1.525 vs 1.56, D4 36.4 vs 36.4), so the instrument
  itself is sound.
- **The policy camera's pose buffers are not maintained.** `data.pos_w` /
  `quat_w_ros` report env 0's spawn pose for every env, so the capture composes
  the camera pose from the robot pose and the configured mount. The camera sits
  at ~0.298 m, not the 0.35 m the nominal mount arithmetic gives, because the
  chassis settles below its reset z.
- **The shared planner emits paths the robot cannot follow.** Its
  line-of-sight shortcut samples segments at half-cell spacing, so a segment
  can graze a cell no sample lands in: 9.5% of a vanilla ProcRoom path's arc
  (4.8% enriched, 0.3–4.6% scanned) sits closer to an obstacle than the
  robot's own inflation radius. Not PR-A's to fix — the planner is a gated
  shared surface — but PR-3a's "promote the path statistics to durable gates"
  should know the floor is not zero.
- **scipy is importable** in the lab interpreter (1.17.0), contradicting the
  design's "verified not available". It is still not a declared dependency, so
  the tool stays numpy-only on policy — and the full EDT the design asked for
  proved unnecessary: clearance is only needed at waypoints, where an exact
  brute-force distance is both cheaper and more accurate than a grid EDT.

## Findings for the operator — all six ruled 2026-07-21

The consult was approved as designed. Each finding below keeps its original
statement of the problem, followed by the ruling that closed it. The rulings
are already propagated into the sections above; this section is the record of
what was decided and why.

**F-1. Byte-identity is certifiable on CPU, not on the GPU.** On CUDA the
Philox generator reserves a per-launch offset block, so a batched draw and
N scalar draws are not the same stream, and `randperm`'s consumption is
size-dependent; the generator has no GPU test at all. The strongest
CI-enforceable claim is *identical CPU output over a seed/difficulty/batch
sweep plus identical per-phase draw structure*. Options: **(a)** accept
that and add a one-off operator CUDA pose-hash before and after PR-1, about
five minutes in any Kit window — *recommended*; **(b)** demand a standing
GPU gate, which needs Kit-window CI that does not exist; **(c)** restate the
constraint as "identical draw sequence" and drop the CUDA obligation.

> **Ruled: option (a).** CPU certification over the seed/difficulty/batch
> sweep plus identical per-phase draw structure is the CI-enforceable
> contract; a one-off operator CUDA pose-hash rides PR-1. No standing GPU
> gate.

**F-2. Golden baseline: current tip, or pre-enrichment?** The enriched
door-width expression is 1 ulp below the pre-enrichment literal, absorbed
everywhere measured. Capturing PR-0's goldens from the pre-enrichment commit
buys parity with history; capturing from the tip silently blesses the
deviation. Recommend the former; the operator owns it.

> **Ruled: pre-enrichment, plus a reconciliation assert** that the current
> tip reproduces those goldens over the full sweep — expectation becomes
> fact. A divergent seed is a STOP that reopens the freeze decision and
> likely motivates making the enriched door-width expression ulp-exact.

**F-3. F2's aperture window is narrow, not empty — but it collides with the
off-path tolerance.** Geometric floor 0.70 m (measured), with raster margin
wanted above it. But `_SUBGOAL_MAX_OFF_PATH_M` is 0.30 m, so below roughly
1.16 m the policy is told a lateral error the aperture does not physically
allow. Options: **(a)** gaps above that collision point — reward-consistent,
but then the aperture statistic must be re-thresholded until it also flags
ordinary furniture gaps; **(b)** narrow gaps with an aperture-aware off-path
tolerance — the honest fix, but it is a shared-reward change needing its own
gate; **(c)** drop F2 and accept the path-topology gap, taking the cheaper
interior-density route instead.

> **Ruled: option (a) for v2 — apertures ≥ 1.2 m**, with PS2 re-thresholded
> so it still counts ordinary furniture-gap threading. **PR-3b unblocks.**
> Option (b) is a shared-reward change filed to v3 beside the parked
> corridor-widening (M2) question it structurally resembles.

**F-4. The D4 target needs a definition.** The gap to the Infinigen band
*floor* is 4.2 points; to the *point estimate* it is 13.5. A noise-floor
measurement **(consult)** puts the unpaired 95% confidence interval at about
±8.3 points at the current sample size — wider than the gap to the floor.
The instruments in PR-A (seeding for paired arms, room-level rather than
pixel-level degrees of freedom) are preconditions, not follow-ups, and the
operator should state which target counts as success. The point estimate is
not reachable by any ordering lever.

> **Ruled: band-floor entry under paired statistics.** Success is the
> paired-measured point estimate ≥ **40.6** with the paired confidence
> interval excluding zero improvement, on PR-A's paired seeding and
> room-level degrees of freedom. The 49.9 point estimate is a direction, not
> a gate — accepted as unreachable by ordering levers, though H1 may move it
> later. The mid-room-column knob therefore ships as a **descriptor-tuned
> probability**, not the measured 99.6% promotion overshoot.

**F-5. The D1 lever contradicts a parked brief.** Reducing spawn erosion is
the largest D1 and D4 lever; the reactive-avoidance brief's M4 wants *more*
spawn standoff so a rear bump is a learnable error rather than an unfair
start state. PR-A takes the D1 side. If M4 is still live, D1 must be fixed
from the geometry side instead, which is weaker and of uncertain sign. This
is a training decision, not a code fact.

> **Ruled: v2 takes the D1 side, keeping the fairness floor.** Reduce the
> spawn erosion as designed, but never spawn in contact range — the part of
> M4's concern that survives is a floor, not a margin (see the D1 section).
> The conflict is recorded in both briefs; M4 re-litigates in v3 against the
> v2 generator baseline, with v2's evidence in hand.

**F-6. Not knowable from the repo.** PhysX contact behavior for two
overlapping kinematic bodies at a compound join (must be measured in Kit);
the per-env cost of the reset path at 256 envs (no benchmark exists at any
env count); and whether any checkpoint has trained against the 8 enriched
goldens — if one has, adding params is a contract break needing new IDs
rather than a re-freeze.

> **Ruled: all three assigned.** (i) Compound-join contact becomes a
> measured-in-Kit gate inside PR-2's validation. (ii) Reset-path cost at 256
> envs is measured by a NOCAM construct-and-step smoke on PR-1 — which this
> design does not otherwise trigger, so PR-1 runs one to host it. (iii)
> **Answered: no training run has consumed any enriched variant** — the v2
> retrain is what this batch precedes, and the NOCAM arms have not run — so
> re-freezing the 8 enriched goldens when params are added is sanctioned by
> their own recorded provenance. Re-verify against run logs at PR time; it
> is cheap.

## Acceptance criteria

- [x] PR-0 lands the guard net against unmodified production code; the
      production diff is empty, the goldens are captured from the
      pre-enrichment tree, and the reconciliation assert shows the current tip
      reproduces them over the full sweep (a divergent seed is a STOP).
      *Landed: 201/201 sweep cases and the 340-op draw sequence reproduce the
      pre-enrichment tree exactly, so the 1-ulp door-width deviation is
      confirmed absorbed. Each obligation was mutation-tested against the
      hazard class it names.*
- [x] PR-A publishes a paired, room-level noise floor for D1 and D4, and the
      re-ranged arm's descriptor table against it — with the spawn fairness
      floor asserted, not assumed.
      *Landed: noise floor ±3.1 points on D4 and ±0.067 m on D1 at 512 rooms
      (95% room-level bootstrap), with a two-seed replicate landing 0.50 points
      / 0.012 m apart. The fairness floor is asserted by a test, and the
      assertion is why the erosion ships at 1 rather than 0 — see the
      measurements section below. Arm A's table is there too, under both
      capture protocols; its effect on D1 and D4 does not resolve at 512 rooms,
      and the instrument findings it produced are a STOP-and-report, not a
      re-rank.*
- [x] The path-statistics tool ships numpy-only and Kit-free, with the
      resolution, inflation-radius and arc-length corrections applied, both
      corpora baselined, and PS2's threshold shown sensitive to
      furniture-gap threading rather than only sub-1.2 m doorways.
      *Landed as `tools/path_statistics.py` + `scripts/measure_path_statistics.py`,
      33 unit tests. The threshold derivation is printed beside the baseline
      with a four-quantile sensitivity sweep. The baseline also corrected the
      comparator the design's PS numbers were measured against — below.*
- [ ] PR-1's surgery leaves every PR-0 golden green with **no edit to any
      frozen literal**, and the per-phase draw witness shows the vanilla
      path unchanged.
- [ ] PR-1 carries the operator-run CUDA pose-hash before and after, and a
      256-env NOCAM construct-and-step smoke reporting reset-path cost.
- [ ] D4 is judged by band-floor entry under paired statistics: point
      estimate ≥ 40.6 with the paired interval excluding zero improvement.
      The column knob ships as a descriptor-tuned probability.
- [ ] Internal walls, if they ship, hold apertures ≥ 1.2 m.
- [ ] The retry ladder's termination bound derives from the park rank's
      length, and protection is implemented as a bounded rank position, never
      an exemption.
- [ ] Any lever that ships names the descriptor or path statistic that gated
      it, and that instrument was shown sensitive to the lever before the
      lever was tuned.
- [ ] No palette growth, no new gym ID, no new composition axis; if any
      becomes necessary, it stops and files a consult.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- **The retrain.** Operator-scheduled; this brief ends at levers
  implemented and descriptor-validated.
- **F4 camera-extrinsics micro-DR and the ceiling probe.** Separate items in
  the v2 batch, independent of placement.
- **H2 mesh-asset curation.** Only if H1 falls short by descriptor, and it
  is palette growth, hence gated.
- **The reactive-avoidance arc.** Task and reward changes, explicitly a v3
  concern — except for the M4 collision recorded in F-5.
- **Any change to the shared proximity reward, the shared planner, or the
  solvability BFS's connectivity.** All are NOCAM-visible; all are gated.
