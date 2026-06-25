# Route start-frame grounding through the strafer_vlm HTTP service

**Type:** refactor / architecture (decouple the VL stack from the Kit env)
**Owner:** DGX agent (`strafer_lab` generator + `strafer_vlm` service)
**Priority:** P2 — the in-process path works for the one-time acceptance run, but it pollutes the canonical `env_isaaclab3` with HF/torch deps and is the wrong architecture for the production corpus capture.
**Estimate:** M (an HTTP grounding-runner backend + a CLI selector + the verdict mapping + tests; the service already exists).
**Branch:** task/grounding-vlm-service-decouple

## Story

As **the DGX operator running the mission-corpus grounding pass**, I want **start-frame grounding to call the existing `strafer_vlm` HTTP service instead of loading Qwen2.5-VL in-process**, so that **the Kit/Isaac-Sim process no longer needs the HF/torch VL stack co-resident — resolving the `cuda-bindings` pin conflict and keeping `env_isaaclab3` clean — and grounding unifies on the same canonical grounder the deployment + finetune lanes use.**

## Motivation

The start-frame grounding seam (the `grounding_frame_provider` + `build_default_grounding_runner`) renders the frame in Kit and then grounds it with an **in-process** `AutoModelForVision2Seq` Qwen2.5-VL load (`build_mission_queue.py`). That forces torch + the HF VL model **co-resident with isaacsim** in one process, which surfaced an **unsatisfiable `cuda-bindings` pin conflict** on the GB10:
- `torch` pins `cuda-bindings==13.0.3`
- `isaacsim-core 6.0.0` pins `cuda-bindings==12.9.4`

To run grounding at all, the acceptance run had to mutate the **shared** `env_isaaclab3` (add `accelerate`, flip `cuda-bindings`) — which also backs training. That coupling is the real cost of the in-process design.

The key insight: only the **render** must be in-process (start poses are RNG-derived mid-traversal, so they can't be pre-rendered to disk). The **VL inference** only needs the rendered frame — it does **not** need to be co-resident with Kit. And `strafer_vlm` **already** wraps Qwen2.5-VL behind a FastAPI service:
- `uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100` (its own `vlm_env`),
- `POST /ground` → `{found, bbox_2d, label}` (bbox in normalized `[0,1000]`).

So route grounding through it: render in Kit, POST the frame to `/ground`, map the response to the verdict.

## Context bundle

- [`mission-generator.md`](mission-generator.md) — the start-frame grounding owner; its real-Qwen gate's metric B.
- `source/strafer_lab/strafer_lab/tools/build_mission_queue.py` — `build_default_grounding_runner` (the in-process runner to make swappable), the `GroundingRunner` interface `(frame, mission_text) -> verdict`, and the verdict plumbing (`yes`/`partial`→ship, cross-room `no`→kept, same-room `no`→`target_not_visible_at_start` re-roll).
- `source/strafer_lab/strafer_lab/tools/grounding_frame_provider.py` — the in-process **render** (stays; pose-dependent).
- `source/strafer_lab/scripts/render_grounded_mission_corpus.py`, `build_mission_corpus.py` — the CLIs to add the backend selector to.
- `source/strafer_vlm/strafer_vlm/service/app.py` + `payloads.py` — the existing FastAPI `/ground` service + its request/response payloads.

## Design

1. **Keep the in-process render.** `grounding_frame_provider` renders the start-frame in Kit, unchanged.
2. **Add an HTTP grounding-runner backend** — `build_http_grounding_runner(endpoint)` matching the `GroundingRunner` interface `(frame, mission_text) -> verdict`. It serializes the PIL frame (base64/bytes per the `/ground` payload), POSTs to the service, and maps the response to the verdict (see §4). No torch / no VL model in the Kit process.
3. **CLI selector** on `render_grounded_mission_corpus.py` (and `build_mission_corpus.py`): `--grounding-backend {http, in-process}` (default **http** for production; `in-process` kept as a fallback) + `--grounding-endpoint` (default `http://localhost:8100/ground`). The factory picks the runner; the seam + verdict plumbing are unchanged.
4. **Verdict mapping** — `strafer_vlm /ground` returns `{found, bbox_2d}` (locate-the-target), the grounding verdict is `yes/partial/no` (visibility). Map `found==true → yes`, `found==false → no`. With the HTTP backend the verdict is binary (no `partial`); the grounding **rate** = found-rate, which is the cleaner signal anyway. Document the mapping; keep `partial` meaningful only for the in-process fallback.
5. **Env outcome** — `env_isaaclab3` no longer needs the HF VL deps (`accelerate`, the torch-VL stack) for grounding; the VL model runs in `vlm_env` behind HTTP, with its own `cuda-bindings`. The pin conflict disappears.

## Acceptance

- [ ] `build_http_grounding_runner` POSTs the rendered frame + prompt to `/ground` and maps `{found}` → `yes`/`no`; unit-tested against a **mocked** HTTP response (no live service, no GPU).
- [ ] `--grounding-backend {http, in-process}` + `--grounding-endpoint` wired into both CLIs; the seam/verdict plumbing in `build_mission_queue.py` is unchanged (the backend only swaps the runner).
- [ ] Operator runbook in the brief: start the `strafer_vlm` service in `vlm_env` (port 8100), then run the grounded corpus with `--grounding-backend http` from `env_isaaclab3` (Kit, no VL deps).
- [ ] The grounded run produces a non-degenerate grounded rate against the live service on ≥1 scene (operator/Kit Phase-2).
- [ ] Brief shipped to `completed/` in the PR per conventions when it lands.

## Out of scope

- The in-process **render** (`grounding_frame_provider`) — stays; pose-dependent, must be in Kit.
- Changing the grounding model, prompt, or the `[0,1000]` bbox convention — reuse the service as-is.
- The detections column / the grounder finetune (separate lanes).
- Reverting the acceptance-run env mutation — that is operator housekeeping after the metric A/B run, independent of this brief.
