# Procedural path-shape mission generator for case-3 training data at scale

**Type:** new feature (sketch — not yet ready to pick up)
**Owner:** DGX agent
**Priority:** P3 (filed-on-trigger; do not pick up until the
trigger condition fires — see below)
**Estimate:** L (~week+; scene-graph annotator + path candidate
enumeration + descriptor synthesis + mission-text templating)
**Branch:** task/harness-procedural-path-shape-generator

## Story

As an **operator who has hit the volume ceiling of
operator-typed path-shape mission demos**, I want **a procedural
generator that walks Infinigen scene metadata + computed
candidate paths and synthesizes path-shape mission text + an
oracle-followable trajectory plan ("go to the chair by hugging
the left wall," "approach the kitchen via the dining room")**,
so that **case-3 training data scales beyond what an operator
can physically type-and-drive in a session, and a future VLA
can learn path-shape generalization rather than memorize a
handful of operator-demonstrated path constraints**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)

Parent design context:
[`MISSION_VALIDATION_ARCHITECTURE.md` §2.3 (case 3 / trajectory shape)](../../MISSION_VALIDATION_ARCHITECTURE.md#section-2--limitations-analysis),
§3.6.a (teleop is the v1 path-shape source; this brief is the
scale-out).

Sibling briefs:
- [`harness-teleop-driver`](harness-teleop-driver.md) — produces
  *operator-typed* path-shape demos (low volume, high quality).
  This brief produces *generated* path-shape demos (high volume,
  lower quality).
- [`harness-oracle-driver`](harness-oracle-driver.md) — consumes
  the generator's `path_constraints[]` output and biases the A*
  cost function to follow the constraint.

## Trigger condition — when to pick this brief up

**Do not pick up until at least one of:**

- A v2 VLA training run on operator-typed path-shape demos has
  shown that path-shape language is in scope but the demo volume
  is the binding constraint (e.g., 50 demos per constraint type
  underfit; 1000 would be expected to fit; operator can't type
  1000).
- A `planner-trajectory-constraint-decomposition.md` (planner-side
  case-3 work) has shipped and the planner now emits structured
  `path_constraints[]` — at which point a matching generator on
  the data side becomes useful.
- A specific downstream brief explicitly requires path-shape
  language at scale.

If none of those have fired, this brief stays parked.

## Architectural sketch

The generator operates entirely on Infinigen scene metadata —
zero sim runs. Output is a `mission_queue.yaml` augmented with
`path_constraints[]` per mission, consumable by either the
teleop driver (operator follows the constraint while driving) or
the oracle driver (A* respects the constraint via a soft cost).

Pipeline:

1. **Scene-graph annotation.** Walk
   `scene_metadata.json`'s `objects[]` + `rooms[]`, plus
   wall geometries derived from Infinigen's USD layer. Build an
   in-memory graph: rooms as nodes, doorways as edges, walls
   tagged with cardinal direction relative to room center.

2. **Candidate-path enumeration.** For each `(start_pose,
   target_object)` pair (drawn from `spawn_points_xy` ×
   `objects[]`), compute multiple A* paths under different cost
   biases:
   - Default cost (shortest path).
   - Wall-following bias (cost penalty proportional to
     distance-from-nearest-wall; produces "hug the wall" paths).
   - Through-room-X bias (cost penalty for any waypoint
     outside room X; produces "via the dining room" paths).
   - Avoid-furniture-Y bias (high cost near specific object;
     produces "around the table" paths).
   - Under-clearance bias (low cost under low-overhang regions;
     produces "under the table" paths if Infinigen scene
     supports the geometry).

3. **Descriptor synthesis.** For each candidate path, label its
   distinguishing features: cardinal walls passed near,
   doorways traversed, regions visited in order, furniture
   passed-around vs. passed-under. Output a structured
   annotation per path.

4. **Mission-text templating.** Map structured annotations to
   natural-language templates with operator-tunable phrasing:
   - `"go to the {target} by hugging the {wall_direction} wall"`
   - `"go to the {target} via the {room}"`
   - `"go to the {target} by going around the {furniture}"`
   - `"go to the {target} by going under the {furniture}"`
   - Multi-constraint compositions: `"go to the kitchen via the
     dining room and hugging the south wall"`.

5. **Augment with 7B Qwen2.5-VL paraphrase.** Same pipeline as
   the harness-expansion brief's mission-text augmentation —
   each templated mission gets N paraphrases for distributional
   diversity.

6. **Output.** `mission_queue.yaml` with rows like:

   ```yaml
   - mission_id: 0123
     target_label: chair
     target_position_3d: [4.2, 1.8, 0.0]
     mission_text: "Go to the chair by hugging the left wall."
     path_constraints:
       - type: wall_follow
         side: south
         max_distance_m: 0.7
     planned_path:
       - [0.5, 0.5]
       - [0.5, 1.5]
       - [1.5, 1.7]
       - [4.2, 1.8]
     paraphrases: ["Approach the chair while staying close to the south wall.", ...]
   ```

The `path_constraints[]` schema is the boundary contract: the
oracle driver consumes it and biases its planner; the teleop
driver shows it to the operator (so the operator knows what
constraint to demonstrate); the v2 VLA training script consumes
it as a structured supervision signal alongside the natural-
language mission_text.

## Acceptance criteria (preliminary; expand at pickup time)

- [ ] **Generator entry point** at
      `source/strafer_lab/strafer_lab/tools/build_path_shape_queue.py`
      consumes `scene_metadata.json` and emits a
      `mission_queue.yaml` with the schema above.
- [ ] **Path-constraint coverage.** For at least 4 constraint
      types (wall-follow, via-room, around-furniture,
      under-clearance), the generator emits ≥ 50 missions per
      type per scene without operator intervention.
- [ ] **Driver consumption.** Both
      [`harness-teleop-driver`](harness-teleop-driver.md) and
      [`harness-oracle-driver`](harness-oracle-driver.md) accept
      the generated `mission_queue.yaml` and respect the
      `path_constraints[]` field — teleop by displaying it to the
      operator; oracle by biasing the planner cost function.
- [ ] **Paraphrase pipeline reuse.** Generator imports the 7B
      Qwen2.5-VL pipeline from
      [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
      Stage 2; doesn't reinvent it.
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5b / 5c gain notes on consuming the generated queue.
      [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `build_path_shape_queue.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Out of scope

- **Generating real-robot path-shape data.** Sim-only — depends
  on Infinigen scene metadata that real-robot deployments don't
  expose at the same fidelity.
- **Validating path-shape constraints at execution time.**
  That's the planner-side `planner-trajectory-constraint-decomposition.md`
  brief — different problem.
- **Free-form natural-language mission generation** (e.g., LLM
  emitting arbitrary path descriptions). Templates + paraphrases
  are the bar here; LLM-from-scratch generation is over-scoped
  and risks hallucinating constraints the scene can't satisfy.
- **Constraint composition beyond ~3 conjunctions.** "Go to
  chair via dining room hugging south wall avoiding table" is in
  scope; arbitrarily-long compositions are not — they break
  templating and the operator's ability to demonstrate them.
