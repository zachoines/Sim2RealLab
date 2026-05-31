# Cognitive-state FSM for `explore_until_visible` (v3)

**Type:** new feature
**Owner:** Either (planner-side FSM prompts + endpoint live on
DGX; executor skill rewrite lives on Jetson — same cross-lane
shape as the v2 scalar-prior brief)
**Priority:** P3 (speculative — only file-on-trigger; v2's
scalar prior is the immediate next step)
**Estimate:** L (~1–1.5 wk; FSM design + per-state prompt design
+ new endpoint + executor skill loop rewrite + A/B eval vs v2 on
the long-horizon mission subset)
**Branch:** task/frontier-cognitive-fsm

**Pickup gate:** Blocked-on-deps until v2 ships
([`llm-guided-frontier-gain`](llm-guided-frontier-gain.md))
**AND** v2 shows a measurable plateau on long-horizon missions
(see "Trigger detail" below). Do not pick up preemptively —
the strafer's current 6 m SLAM horizon may not yet stress the
regime where CogNav's FSM beats LFG's scalar prior, and v2 is
strictly simpler to maintain.

## Story

As an **operator running cold-start cross-room missions that
span 3+ rooms or require re-verification of mis-grounded
sightings — the regime where v2's per-step scalar prior keeps
visiting plausible-but-wrong frontiers before recovering — I
want **the frontier-exploration loop to switch among cognitive
states (broad exploration → contextual exploration →
identification → verification) under LLM control**, so that
**mission success rate and time-to-target on long-horizon
multi-room missions improve measurably versus v2, especially
when initial frontier choices need to be revisited or
re-grounded**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`llm-guided-frontier-gain`](llm-guided-frontier-gain.md) —
  v2, the predecessor. v3 replaces v2's single per-attempt
  ranking call with a state-machine driver that can call the
  LLM multiple times per step. The geometric detector and the
  bounded-termination scaffolding survive both v2 and v3.
- [`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md)
  — v1, the substrate. The detector, costmap snapshot
  plumbing, and skill registration are unchanged across all
  three versions.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — supplies the same room-membership / connectivity signals
  v2 consumes; v3's per-state prompts also depend on them.

## Trigger detail (un-park conditions)

v3 is filed-on-trigger because the brief author's prior is that
v2's scalar prior captures most of the strafer-regime gain. File
this brief if v2's post-ship A/B eval shows **at least one** of:

- v2 fails on ≥ 30% of cold-start missions spanning ≥ 3 rooms
  (the long-horizon regime where CogNav's verification state
  matters most).
- v2 wastes ≥ 2 frontier visits on average for missions where
  the LLM's first ranking was geometrically reasonable but
  semantically wrong — the failure mode where v2 cannot
  recover because there is no "re-verify and re-rank from a
  new observation" state in the loop, only a flat retry.
- Real-robot deployment surfaces hard cases where a confidently
  wrong pre-grounded VLM sighting drives the robot into a wrong
  room and v2 cannot back out before the mission deadline.

If none of these bite within ~3 months of v2 shipping, the
likely correct action is to **delete this brief**, not pick it
up. v3's complexity is only justified by evidence v2 plateaus
at a meaningful boundary; absent that evidence, the parked
brief is just speculation tax.

## Context

### Why a state machine, not a richer scalar prior

The v2 brief's "Design: scalar prior, not state machine" section
already enumerates the trade. A few things change in v3 that
v2's prior cannot capture:

- **Per-step *behavior*, not just per-step *ranking*.** v2 picks
  the highest-scoring frontier and runs the same nav+scan
  routine every step. CogNav's "identification" state runs a
  denser scan with multiple grounding attempts; "verification"
  state revisits previous candidate rooms to confirm or
  invalidate; "broad exploration" deprioritizes the LLM in
  favor of pure geometric gain when no semantic context is
  available yet. The strafer cannot express any of these from
  a multiplicative prior alone.
- **Memory across steps.** v2 treats each step's LLM call as
  stateless (other than the persistent visited-set). v3's FSM
  carries explicit history into prompts ("you ruled out the
  bedroom 3 steps ago; the kitchen is still unverified") so
  the model doesn't re-derive the same mistake.
- **Failure recovery is a state, not a fallback.** v2's only
  recovery from a bad rank is to mark the frontier visited and
  re-rank. v3's verification state explicitly re-grounds from
  multiple poses before committing — the failure mode v2 cannot
  fix.

### Reference architecture (CogNav, ICCV 2025)

The paper's FSM has four states:

| State | Trigger to enter | Per-step behavior | Trigger to leave |
|---|---|---|---|
| Broad exploration | Cold start; no semantic anchors | Geometric gain dominates; LLM optional | First semantic anchor identified that matches target context |
| Contextual exploration | Target-room candidates exist in the LLM's prior | Per-frontier LLM scoring drives ranking (≈ v2 behavior) | Candidate sighting found in grounding |
| Identification | Candidate sighting from contextual exploration | Dense scanning at the candidate pose; multiple grounding attempts at different headings | Either: sighting confirmed (→ verification), or rejected (→ contextual exploration) |
| Verification | Sighting confirmed; about to commit | Approach the target; re-ground at decreasing standoff; bail if confidence drops | Confidence threshold met (→ exit success) or confidence collapses (→ contextual exploration) |

Each step, the LLM observes `(target, current_scene, history,
current_state)` and emits either a transition or an action
within the current state. Per the paper, all LLM calls are
zero-shot — no fine-tuning. Two to three calls per step
(state classifier + behavior selector); 3–5× the per-step
latency of v2.

### What v3 changes from v2

| Component | v2 → v3 change |
|---|---|
| `frontier.py:detect_frontier_clusters` | Unchanged. |
| `FrontierCluster`, `CostmapSnapshot`, costmap plumbing | Unchanged. |
| `rank_frontiers` + its `llm_prior_fn` seam | Vestigial — used only inside the "contextual exploration" state. Kept for the state's reuse. |
| `_explore_until_visible` body | **Rewritten** as an FSM driver: each tick, query the planner for `(state, action)`, dispatch the action, update history, loop. |
| Bounded termination (max_frontiers, timeout_s, visited-set) | Survives — state-aware budgets layered on top. |
| Skill registration in `DEFAULT_AVAILABLE_SKILLS` | Unchanged. The skill name and dispatch survive — v3 is internally different but externally drop-in. |
| Planner endpoint | New: `POST /step_navigation` returning `(state, action)` per tick. The v2 `POST /rank_frontiers` endpoint stays available and is called from inside the contextual-exploration state — v3 composes with v2 rather than replacing it. |
| Mission history | New — v3 needs a per-mission history buffer the FSM driver can pass into prompts. Lives in `_MissionRuntime` or a parallel struct. |
| Tests | Detector tests survive; v2 mock-LLM tests survive as the contextual-exploration state's tests; new FSM-driver tests are the bulk of new test surface. |

Rough survivorship from PR #40 (v1) into v3: ~60% of lines. The
detector, snapshot, and skill registration are stable; the loop
body is rewritten.

### Exploration legs disarm the CLIP tripwire — by leg-type

The FSM's identification and verification states **revisit
candidate rooms and re-ground from multiple poses by design** —
they deliberately leave the geodesic line to the mission target
even more than v2's flat per-step ranking does. The CLIP
mid-mission tripwire from
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
must therefore be **disarmed on exploration legs by leg-type**, so
that none of these states is scored as deviation. The per-leg
deviation contract — which leg-types arm the tripwire and which
disarm it — is owned by
[`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md);
this brief does not re-encode that table, it only relies on the
exploration / re-verification leg-types being on the disarmed side
of it.

Frontier-vs-target consistency — including the verification
state's re-grounding decision — stays a **soft LLM ranking**
(this skill inherits v2's `llm_prior` gain). It biases which
frontier or candidate room the FSM investigates next; it is
**never a hard CLIP abort**.

### Latency budget

v3 fires the LLM 2–3× per exploration step vs. v2's 1×. Each
exploration step is ~10–30 s of robot work, so the per-step
LLM budget of 1–2 s in v2 grows to 3–6 s in v3 — still well
under the per-step ceiling.

### Failure handling

Every failure mode falls back to v2 behavior:

- FSM endpoint timeout / 5xx → v2 contextual-exploration step.
- LLM returns an invalid state name → v2 contextual-exploration step.
- State-machine loops between two states for ≥ N transitions
  without progress → v2 contextual-exploration step + warn.

The operator-side kill switch (`fsm_enabled = false`) is the
final fallback — recovers v2 exactly.

## Acceptance criteria

- [ ] **FSM design documented.** The four-state graph + per-state
      behaviors + transition triggers live as a Mermaid diagram in
      [`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      or a sibling design doc, so the next reader can audit the
      contract without reading the brief.
- [ ] **Endpoint shipped.** `POST /step_navigation` on the DGX
      planner service, documented under
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      with input/output schema.
- [ ] **Schema additions.** `StepNavigationRequest`,
      `StepNavigationResponse`, `FsmState`, `FsmAction` added
      to the schemas package.
- [ ] **FSM driver in the skill.** `_explore_until_visible`'s
      body becomes a state-machine driver; per-state behaviors
      are factored as helper methods so each state is unit-testable
      in isolation.
- [ ] **v2 composition.** The contextual-exploration state calls
      `POST /rank_frontiers` from v2 unchanged — v3 builds on v2's
      endpoint rather than duplicating it.
- [ ] **`fsm_enabled = false` recovers v2 exactly.** A regression
      test pins this: with the FSM disabled, plan outputs and
      frontier-visit order are byte-identical to v2 fixtures.
- [ ] **Latency budget.** Mean total per-step LLM latency
      (sum across FSM calls) on the mission-generator long-horizon
      eval set is ≤ 6 s; p95 ≤ 10 s. Measured and reported in the
      PR description.
- [ ] **A/B eval.** On at least 20 long-horizon (≥ 3 rooms)
      cold-start missions, v3 shows ≥ 15% improvement in
      time-to-target OR ≥ 1 fewer frontier-visit on average vs. v2
      on the same missions. Pre-register the primary metric.
      No regression vs. v2 on near-target / single-room missions.
- [ ] **No regression** in v2's failure modes. With any FSM
      failure-handling fallback firing, behavior is identical
      to v2.
- [ ] **Runtime-legal inputs only.** A grep of the new code for
      `scene_metadata`, `scene_labels`, `room_adjacency` returns
      zero hits.
- [ ] **Doc surface updates.**
  - [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md):
    endpoint table gains a `POST /step_navigation` row.
  - [`docs/STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../../STRAFER_AUTONOMY_NEXT.md#1101-known-limitation-multi-room-navigation):
    update the variant list — unguided (v1), LLM-guided
    scalar prior (v2), cognitive FSM (v3).
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- v2 skill structure: [`llm-guided-frontier-gain`](llm-guided-frontier-gain.md).
  v3 reuses v2's `POST /rank_frontiers` endpoint inside the
  contextual-exploration state — read v2's prompt design first.
- v1 frontier detector + snapshot plumbing:
  [`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md).
- Reference papers:
  - **CogNav, ICCV 2025**
    ([PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_CogNav_Cognitive_Process_Modeling_for_Object_Goal_Navigation_with_LLMs_ICCV_2025_paper.pdf))
    — the FSM design this brief follows. Read §3 (cognitive
    process modeling) and §4 (per-state prompt design) at
    minimum.
  - **LFG (Language Frontier Guide)** (arXiv:2310.10103) —
    v2's scalar-prior predecessor. Read its §5 "comparison to
    state-machine baselines" if relevant.
  - **FSR-VLN** (arXiv:2509.13733) — hierarchical scene-graph +
    LLM reasoning. Informs the prompt context for v3 just as it
    informed v2.
- LFG-style sampled-rationale variance reduction: if v2 already
  used `n_samples` rationale averaging, v3's per-state prompts
  inherit the same pattern.

## Out of scope

- **Online LLM fine-tuning** on FSM trajectory outcomes. The
  v2 brief's same exclusion applies; pursued only inside the
  experimental epic's VLA work.
- **Multi-agent / collaborative exploration.** CogNav covers
  single-agent; the strafer's MVP is single-agent; no
  multi-agent extension is contemplated.
- **Open-vocabulary state machines.** v3 commits to the
  four-state model from the CogNav paper. Hyperstate variants
  (more granular states, hierarchical FSMs) are deferred to a
  v4 if v3 itself plateaus.
- **Replacing v2.** v3 composes with v2 (v2's endpoint is called
  from v3's contextual-exploration state). If v3 is later
  retired, v2's surface still works.
- **Training-time `scene_metadata.json` consumption.** Out of
  scope and explicitly forbidden — same sim-to-real rule as
  the v1 and v2 briefs.
