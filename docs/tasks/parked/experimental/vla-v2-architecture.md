# Spin up a v2 end-to-end VLA architecture as a toggleable autonomy backend

**Type:** investigation / research / new feature
**Owner:** Either (the heart of the work is DGX-side — VLA service,
training, eval; the executor-side change to route missions to the
new backend is a small Jetson-lane edit)
**Priority:** P3 (research direction; depends on
[`harness-architecture`](../../active/harness/harness-architecture.md)
Tier 1 (teleop, primary data path) + Tier 2 (bridge, supplement)
shipping to provide the action-labeled LeRobot v3 corpus;
explicitly *additive* to the v1 MVP — does not block any
in-flight feature)
**Estimate:** XL (~multi-week; new service + training pipeline +
eval methodology + a realistic-but-research-flavored success bar,
not a production ship)
**Branch:** task/strafer-vla-v2-architecture

## Story

As an **operator who wants hands-on Vision-Language-Action
experience and a v2 dual-system architecture for `strafer` that
sits alongside the existing tiered MVP**, I want **a toggleable
end-to-end VLA backend (planner + executor + skill abstractions
collapsed into one model that consumes mission text + perception
and emits low-level commands) plus the training pipeline and
sim-eval methodology to realistically spin it up against the
Infinigen harness**, so that **the project gains a concrete
research vehicle for end-to-end VLAs without disturbing the v1
MVP, with falsifiable acceptance criteria calibrated to a
research outcome rather than a production ship**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md),
specifically §3.3 (small VLA — integration depth, model choices,
Orin Nano feasibility), §3.6 (MVP-as-teacher distillation), §4.2
(why no from-scratch VLA in isolation).

Retired sibling brief — a smaller VLA-as-validator on-ramp that
was considered and explicitly retired in favor of this end-to-end
research arm + the cascade-improvements path:
[`completed/learned-mid-mission-validator.md`](../../completed/learned-mid-mission-validator.md).

Map-conditioning contract (decides whether/how this VLA reads
the semantic map):
[`vla-v2-map-conditioning`](vla-v2-map-conditioning.md) — picks
one of {serialize the symbolic regions / consume the implicit
memory map / no consumption} via a three-row ablation. **Read it
before adding any map input to this VLA's `(frame, text,
action)` tuple** — the conditioning shape is decided there, not
here. Default (this brief, no map conditioning) is the
ablation's baseline.

## Context

### Sim-first by design

The v2 path is **sim-only until a more capable robot compute
upgrade exists.** Real-robot deployment requires a Jetson capable
of co-tenanting a small VLA alongside Nav2 + RTAB-Map without
thrashing memory or latency budgets — Orin Nano (8 GB unified)
plausibly fits Octo-base only, and only after Nav2 / RTAB
profiling on a populated scene. This brief produces the **sim
research vehicle**; real-robot transfer is filed as a separate
brief once a Jetson upgrade arrives or the model footprint
shrinks sufficiently. Treat the v2 path as a research arm of
the project, not a deployment track.

### Why now

The v1 MVP (`strafer_lab` RL controller + Nav2 + `strafer_autonomy`
planner + executor + `strafer_vlm` grounding) is a tiered hybrid
architecture: each tier has explicit intermediate representations
(JSON plans, skill calls, costmap paths). It's inspectable but
brittle at the boundaries — case 2 / case 3 from
[`MISSION_VALIDATION_ARCHITECTURE.md` §2.3](../../../MISSION_VALIDATION_ARCHITECTURE.md#section-2--limitations-analysis)
exemplify what falls between the cracks.

A v2 end-to-end VLA architecture takes the opposite design stance:
collapse planner + executor + skills + control into one model that
ingests mission text + RGB-D + odom + (optionally) prior actions and
emits continuous commands. **It trades inspectability for
gradient-flow-end-to-end** and exposes the project to the modern
VLA training stack.

This brief explicitly aims at *research experience*, not a production
replacement. The v1 MVP keeps shipping; the v2 backend is an
opt-in alternative behind an env-var toggle.

### Hypothetical v2 architecture

Component-level changes vs. v1, with toggle:

```
v1 (current, default):
  mission text
    → strafer_autonomy.planner LLM (DGX:8200)
    → JSON plan
    → strafer_autonomy.executor (Jetson)
    → skill calls (scan_for_target / project / navigate_to_pose / verify_arrival)
    → strafer_vlm grounding (DGX:8100)        Nav2 (Jetson)
    → RL controller (Jetson)
    → /cmd_vel → safety filter → wheels

v2 (new, opt-in):
  mission text + RGB-D + odom + prior actions
    → strafer_vla service (DGX:8300)
    → action tokens → de-tokenizer
    → /cmd_vel → safety filter (Nav2 costmap stays as velocity-clamp) → wheels
```

The VLA service mirrors the existing `strafer_vlm` pattern: HTTP
on the DGX, called by the Jetson executor over LAN, behind a
single-worker queue. **Nav2's costmap stays in the loop as a
safety filter** — the VLA emits `cmd_vel`, the costmap layer
clamps based on lethal cells, then the command reaches the
wheels. Hard-coding pure VLA-to-wheels is out of scope until a
separate safety study justifies it.

**The toggle.** A new env var
`STRAFER_AUTONOMY_BACKEND={skills, vla}` (default `skills`)
selects which path the executor uses. When `vla`, the executor
bypasses planner + skills and instead streams perception to the
VLA service at the configured rate (5–10 Hz). The
`execute_mission` action contract on the Jetson side is
unchanged, so the harness, CLI, and operator clients work
against either backend without code changes.

### Where the VLA runs

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **DGX-side service (default for this brief)** | Full Blackwell GPU; can run Octo-base / TinyVLA-mid / OpenVLA-7B locally; no Jetson co-tenancy pressure with Nav2 / RTAB; mirrors how `strafer_vlm` works today. | LAN round-trip latency (~10–50 ms over the existing CycloneDDS cross-host setup); the bridge becomes a real-time critical path during `vla` mode. | **Start here.** |
| **Jetson-side service (future)** | No LAN round-trip; matches the long-term "deploy on the robot" vision. | Only Octo-base (~50–80 ms FP16) realistically fits alongside Nav2 + RTAB; OpenVLA / NaVid / TinyVLA-large don't. Co-tenancy memory budget is tight (8 GB unified). | **Out of scope; future brief once a Jetson-fitting checkpoint exists.** |

### Model choice — deferred to a selection rubric

The brief deliberately does **not** pre-commit to a base model.
Run the rubric below as the first phase of the work; document the
chosen model and the score; carry forward.

**Selection rubric.** Score each candidate on these axes; a
weighted total picks the v2 reference model:

| Criterion | Weight | What "good" looks like |
|---|---|---|
| Public training code + standard dataset format | High | Documented training loop (RLDS / HF datasets); reproducible on the DGX from a public checkpoint. A documented loop is more valuable than 10× the parameter count for this brief's learning goal. |
| Sim-fine-tune-friendly recipe | High | Paper or maintainer-blessed example showing sim-only fine-tuning (Habitat / Isaac / RLBench / similar). VLAs trained exclusively on real-robot demos often resist sim adaptation. |
| Mecanum-holonomic action-space adapter feasibility | High | Architecture exposes the action head clearly so the strafer-specific `(vx, vy, ωz)` head is a contained change, not a fork of the model. |
| Public sim benchmark numbers | Medium | Reported success rates on a sim benchmark provide a sanity baseline for whether your numbers are plausible. |
| DGX fine-tune memory footprint | Medium | Can the chosen variant fine-tune on a single Blackwell without aggressive sharding or quantization gymnastics? |
| Long-term Jetson-migration path | Low (v2), high (v3) | Does the model have a small variant that *could* fit Orin Nano alongside Nav2 + RTAB? Don't gate v2 on this, but record it. |

**Candidates to score (non-exhaustive; evaluate at brief-execution
time, since the VLA literature moves fast):**

| Model | Params | DGX fit | Action space | Public training code? |
|---|---|---|---|---|
| Octo-base | 27 M | Trivial | Generalist, RLDS-trained | Yes ([github.com/octo-models/octo](https://github.com/octo-models/octo)) |
| Octo-small | 27 M | Trivial | Same | Yes |
| OpenVLA-7B | 7 B | Comfortable | LIBERO + Open X-Embodiment | Yes ([github.com/openvla/openvla](https://github.com/openvla/openvla)) |
| TinyVLA / MobileVLA-end | 80–500 M | Trivial | Varies | Partial |
| π0 / π0.5 | ~3 B | Comfortable | Mobile manipulation | Partial (Physical Intelligence's release) |
| NaVid / NaVILA | 7–8 B | Comfortable on DGX | VLN-flavored | Partial |

The chosen model + rubric scores get recorded in the brief's
sim-eval report and become part of the project's lessons-learned
trail. If the rubric picks a model that later proves a dead end,
the trail makes that legible.

### Training pipeline

End-to-end, all DGX-side. Step 1 is **not** in scope for this
brief — data capture ships separately. This brief assumes those
upstream briefs have shipped and the harness output is
behavior-cloning-grade.

1. **Data harvest (out of scope; prerequisite briefs).** The
   harness emits per-tick `(frame, depth, pose, last_cmd_vel,
   mission_text, mission_id, …)` records via the LeRobot v3
   schema defined in
   [`harness-architecture`](../../active/harness/harness-architecture.md).
   **Default data source: teleop demos** (per
   [`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.a](../../../MISSION_VALIDATION_ARCHITECTURE.md#36a-teleop-demos-primary-canonical))
   captured by harness
   [Driver: teleop](../../active/harness/harness-architecture.md#driver-teleop) —
   this matches how every published wheeled VLA is trained.
   Bridge-driver data
   ([Driver: bridge](../../active/harness/harness-architecture.md#driver-bridge))
   is a permissible *supplement* once the v1 stack is reliable
   (§3.6.b). Future scale-out via the scripted driver
   ([Driver: scripted](../../active/harness/harness-architecture.md#driver-scripted))
   if teleop throughput becomes the bottleneck (§3.6.c).
   Strafer is **mecanum holonomic**, not differential drive,
   which is unusual relative to most VLA training data — the
   action-space adapter for the chosen model is the first
   technical risk (see below).

2. **Dataset assembly.** Convert harness output to **RLDS**
   (Octo's native format) or HuggingFace datasets with a flat
   schema. Tooling lives at
   `source/strafer_lab/strafer_lab/tools/build_vla_dataset.py`,
   mirrors how `dataset_export.py` and `prepare_vlm_finetune_data.py`
   work today.

3. **Action-space adapter.** Decide one:
   - **Discrete bin tokenization.** 256 bins per axis × 3 axes
     = ~768-token vocabulary. Standard for OpenVLA / RT-2.
     Loses precision near zero (most strafer commands are
     small).
     - Refinement: non-uniform bins (denser near zero) or
       log-spaced.
   - **Action diffusion head.** Continuous output via a
     diffusion decoder. Standard for Octo, π0. Better
     precision; harder to train stably on small data.
   - **Custom strafer head: continuous regression on
     `(vx, vy, ωz)` with a small MLP.** Simpler, less
     "VLA-canonical." Good fallback if the canonical recipes
     overfit.

   The choice is sticky — switching later is a from-scratch
   retrain. The brief should pick early and document why.

4. **Pretrained init.** Don't train from scratch. Load Octo-base
   (or OpenVLA-7B) weights, swap the action head for the
   strafer-specific one, freeze the visual + language towers,
   train only the head + final fusion layers for a
   first-pass run. Unfreeze later if the head plateaus.

5. **Training loop.** SFT on the harness's
   `(frame_sequence, mission_text, action_sequence)` tuples,
   filtered to `mission_state=succeeded, reachable=True`
   missions only (positive-only behavior cloning). AdamW,
   conservative LR (1e-5 to 1e-4 depending on which layers are
   unfrozen), MLflow tracking via the same
   `--mlflow-experiment` flag pattern that
   [`finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py)
   uses.

6. **Optional DPO/RLHF on outcomes.** Once SFT is stable, use
   `(reachable=True)` vs. `(reachable=False)` mission pairs as
   preference data. DPO is cheaper than full RLHF; either is
   research-grade scope.

7. **Sim-eval.** The harness drives the v2 backend via the
   toggle and emits its LeRobot v3 dataset per
   [`harness-architecture`](../../active/harness/harness-architecture.md).
   Compare outcome metrics (reachability, time-to-arrival,
   false-stop / false-continue rates) against the v1 baseline
   on the same scenes; outcome labels live in the per-episode
   `outcome` column under `meta/episodes/`.

### Technical considerations and risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Action-space mismatch.** Mecanum holonomic `(vx, vy, ωz)` is rare in VLA training data. Pretrained weights expect differential-drive or 6-DoF arm action distributions. | High | Custom action head trained from scratch; freeze the upstream towers. Document the chosen tokenization explicitly. |
| **Sim-to-real gap.** Pure-sim training on Infinigen → real-robot deployment will fail without a transfer step. | High | Out of scope for this brief — sim-eval only. A future brief layers in real-robot fine-tuning once sim-eval looks plausible. |
| **Data volume.** Hundreds of missions per scene is small relative to OpenVLA's training set (~970k trajectories) or Octo's pretrain corpus. | Medium | Plan to *adapt*, not retrain. Use frozen-tower training; mix in publicly-available navigation datasets (Habitat, R2R-CE) as auxiliary if data scarcity bites. |
| **Mission-language distribution.** Current harness emits short endpoint-shaped texts ("go to the chair"). Path-shape and multi-step language are out of distribution. | Medium | Augment harness mission_text generator to include 7B-Qwen-paraphrased variations (already used for descriptions in Stage 2 of `generate_descriptions.py`). |
| **No `verify_arrival`.** v2 has no explicit arrival check — the VLA has to decide it's done. | Medium | Add a "stop" action token; train on harness `mission_state=succeeded` examples ending with a stop. Eval the false-stop and false-continue rates separately. |
| **Recovery semantics.** The v1 executor has explicit skill-level recovery (re-scan, re-ground, retry). v2 has none unless the VLA learns it. | Medium | The harness can inject deliberately-confused mid-leg observations to force recovery learning. Or: keep a v1 watchdog (mission timeout, costmap-stuck detection) that aborts the v2 mission and returns control to v1. |
| **Co-tenancy.** DGX-side runs alongside `strafer_vlm` (Qwen2.5-VL-3B at port 8100), `strafer_autonomy.planner` (LLM at port 8200), Isaac Sim bridge, and now the VLA service at port 8300. | Low–Medium | Each service uses separate GPU memory (DGX has 128 GB unified). Cap each service's memory at startup; document via `nvidia-smi` snapshot. |
| **Real-time critical path.** When `STRAFER_AUTONOMY_BACKEND=vla`, the LAN becomes real-time critical — VLA latency directly drives control rate. | Low (DGX-side) | Existing LAN bench numbers (`bridge-runtime-invariants.md`) put cross-host DDS at <50 ms p99. HTTP add overhead similar. Cap VLA per-call budget at 200 ms; alert the executor if exceeded. |
| **Inspectability regression.** Operator can no longer see why a mission did what it did. | Inherent | Log VLA action-token stream + per-step VLM `/describe` snapshots to a per-mission trace. Inspectable post-hoc, not at decision time. |

### What this brief produces

Five concrete artifacts:

1. **`source/strafer_vla/`** — new Python package mirroring
   `strafer_vlm`'s structure: `service/` (FastAPI app),
   `runtime/` (model loading + inference), `training/`
   (fine-tune script + dataset adapter), `tools/` (CLI smoke
   tests), `tests/`.
2. **DGX HTTP service** at `:8300` with endpoints `/health`,
   `/act` (mission_text + RGB(D) + odom → action tokens +
   decoded velocity).
3. **`strafer_autonomy.clients.vla_client.HttpVLAClient`** —
   parallel to `vlm_client.HttpGroundingClient`. DGX-lane file
   per
   [`ownership-boundaries.md`](../../context/ownership-boundaries.md).
4. **`STRAFER_AUTONOMY_BACKEND` env-var toggle** in
   `executor/main.py` and a new `MissionRunner.run_via_vla()`
   path that bypasses planner + skills. Jetson-lane.
5. **Training pipeline + sim-eval report** under
   `docs/artifacts/vla_v2/<run_id>/` summarizing the chosen
   model, action-space recipe, dataset stats, sim-eval metrics
   vs. the v1 baseline, and lessons learned.

## Acceptance criteria

This is a research brief. The bar is **a working v2 backend with
honest measurement**, not "v2 beats v1."

- [ ] **`strafer_vla` package builds and tests pass** (`pip install
      -e source/strafer_vla; pytest source/strafer_vla/tests/`).
- [ ] **DGX service runs end-to-end** — `make serve-vla` (or
      equivalent) starts the FastAPI app on port 8300; `curl
      :8300/health` returns `model_loaded: true` for the chosen
      checkpoint.
- [ ] **Toggle works** — `STRAFER_AUTONOMY_BACKEND=vla` plus a
      single mission via `strafer-autonomy-cli submit` reaches the
      VLA service, returns action tokens, and drives the
      simulated robot at least one step. Captured as a one-line
      log excerpt in the PR description.
- [ ] **Training pipeline runs end-to-end** on at least 30 harness
      missions across ≥ 2 Infinigen scenes. Output: a checkpoint
      + ONNX export under `~/.strafer/models/vla/`.
- [ ] **Sim-eval report** at
      `docs/artifacts/vla_v2/<run_id>/report.md` covering:
  - Chosen model + action-space recipe + why.
  - Dataset stats (mission count, total frames, action
    distribution, success/failure split).
  - Per-mission outcomes: reachability rate, time-to-arrival,
    false-stop / false-continue rates.
  - Side-by-side comparison vs. the v1 baseline on the *same
    harness episode set*.
  - At least three named failure modes with frame examples.
  - Lessons learned section: what surprised, what was harder
    than expected, what's worth carrying forward.
- [ ] **A short addendum to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)**
      records the v2 results and recommends one of: continue
      (file follow-up briefs for Jetson migration / real-robot
      transfer / better data); pause (defer until §3.1 / §3.5
      cascade ships); retire (architecture is a dead end —
      document why).
- [ ] No regression in the v1 stack with
      `STRAFER_AUTONOMY_BACKEND=skills` (default). Smoke this in
      the PR description.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- Octo training code: [github.com/octo-models/octo](https://github.com/octo-models/octo).
  Public RLDS converters; existing wheeled-base adaptations.
- OpenVLA training code: [github.com/openvla/openvla](https://github.com/openvla/openvla).
  Heavier; documented LoRA fine-tune recipe.
- HuggingFace cache on the DGX already has Qwen2.5-VL-3B and
  Qwen3-4B (verify via `ls ~/.cache/huggingface/hub/`).
  Octo / OpenVLA weights need a separate fetch.
- The bridge's perception camera is 640×360
  ([`test_d555_perception_cfg.py:50`](../../../../source/strafer_lab/test/sensors/test_d555_perception_cfg.py#L50)).
  VLA visual towers usually expect 224² or 336²; preprocessing
  follows the
  [§2.2 letterbox-vs-center-crop discussion](../../../MISSION_VALIDATION_ARCHITECTURE.md#22-structural-the-bandwidth-lost-between-perception-camera-and-clip-input).
- The mirroring service to model after:
  [`source/strafer_vlm/`](../../../../source/strafer_vlm/). Same
  FastAPI + `ThreadPoolExecutor` + health-check pattern.
- The harness data path:
  [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
  Stages 5 + 6.
- VLA architectural references already cited in
  [`MISSION_VALIDATION_ARCHITECTURE.md` §3.3](../../../MISSION_VALIDATION_ARCHITECTURE.md#33-small-vla):
  RT-2, OpenVLA, Octo, π0, NaVid, NaVILA, MobileVLA, TinyVLA.
- For the safety filter: existing Nav2 costmap layer that the
  velocity passes through. The brief picks the integration
  point — directly upstream of the wheels driver, downstream of
  Nav2's `local_costmap`.

## Out of scope

- **Real-robot deployment.** Sim-eval only. A future brief
  layers in real-world transfer once sim-eval is meaningful.
- **Jetson-side VLA service.** DGX-only for this brief. Future
  brief once a Jetson-fitting checkpoint exists and Nav2 / RTAB
  co-tenancy is profiled.
- **Replacing the v1 stack.** v2 stays opt-in. v1 remains the
  default and the production target.
- **Trajectory-shape language ("hug the wall," "via the dining
  room").** Family 3 work — separate brief if pursued. v2 is
  Family 2 (end-to-end actions).
- **(Was) Multi-room navigation deferral.** No longer applies —
  multi-room is the MVP default per
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  (which lifts §1.10.1) and
  [`scene-connectivity-validation`](../../active/multi-room/scene-connectivity-validation.md).
  This brief's training corpus is multi-room by default.
- **From-scratch VLA training.** Adapt an existing checkpoint
  only.
- **Retiring `strafer_vlm` or the planner.** Both stay running;
  the v1 stack still depends on them. v2 is *additive*.
- **Replacing the safety filter.** Nav2's costmap layer stays
  in the loop as a velocity clamp. A future brief may reduce
  this to an end-to-end safety-aware VLA, but not here.
