# Move the mission generator's text LLM out of the Kit process (reuse serve-planner)

**Type:** refactor / deployability follow-up
**Owner:** DGX agent (strafer_lab side) — the new HTTP endpoints are
cross-lane work the `strafer_autonomy` lane owns; coordinate.
**Priority:** P3 (non-blocking quality / deployability follow-up. The v1
corpus ships fine with oracle waypoints + REG-disambiguated templated
text, which cannot hallucinate. The LLM's only added value is the
naturalness / diversity of phrasing; it is **not** capture-blocking.)
**Estimate:** M (small `strafer_lab`-side HTTP-client runners + CLI
wiring + a stubbed unit test; the bulk is the cross-lane endpoint
addition on the planner service, which already loads the model)
**Branch:** task/out-of-process-mission-text-llm

## Story

As a **DGX operator who wants naturalistic, diverse LLM-authored mission
text (or a grounded corpus run that uses the LLM waypoint planner)
without re-introducing the torch-vs-Isaac-Sim `cuda-bindings` conflict
the geometric grounding pivot removed**, I want **the mission
generator's text passes (waypoint planner + paraphrase) to run
out-of-process behind the existing `serve-planner` HTTP service instead
of loading `Qwen3-4B` inside the Kit process**, so that **an in-Kit
grounded corpus run can use real LLM text without flipping the canonical
isaaclab env's `cuda-bindings` pin, while the v1 oracle + templated
corpus keeps shipping unblocked**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/planner-request-schema.md`](../../context/planner-request-schema.md) —
  the existing `serve-planner` request/response contract this brief
  extends with text-generation endpoints.

Parent / sibling briefs:
- [`mission-generator`](../../active/harness/mission-generator.md) —
  the generator this serves. Its **grounding** pivot (VL → geometric)
  removed the *VL* model from Kit; this brief closes the same gap for
  the *text* model. The generator's `build_default_waypoint_runner` /
  `build_default_paraphrase_runner` are the seam this brief replaces.
- [`mission-text-enrichment`](../../completed/harness/mission-text-enrichment.md) —
  the REG disambiguator + templated paraphrases that make the
  model-free corpus shippable without an LLM. This brief is the
  *quality lift* over that templated baseline, not a replacement for it.

## Context

### Why this is a follow-up, not a blocker

The mission generator ships a usable corpus with **no LLM in the loop**:
the oracle planner produces clean shortest-path waypoints, and
`mission_text_builder.disambiguate` + templated paraphrases produce
groundable, REG-disambiguated mission text that *cannot* hallucinate (it
only names objects the scene actually contains, with anchors the camera
can ground). The LLM waypoint planner and paraphrase model add
naturalness and phrasing diversity — a training-data quality lift — but
nothing the v1 capture depends on. So this is filed parked,
filed-on-trigger: pick it up when LLM-quality mission text is wanted over
oracle + templated, **or** when a grounded run needs `--use-planner-llm`
/ `--use-paraphrase-llm` and must not flip the env's `cuda-bindings` pin.

### The problem: the text LLM is the same in-Kit conflict the VL pivot removed

The grounding pivot (`mission-generator`, geometric gate) took the *VL*
model out of the Kit process — visibility is read off Replicator
annotators against the known target, so no torch-VL model is co-resident
with Kit, and the torch-vs-Isaac-Sim `cuda-bindings` conflict is gone for
the grounding pass.

The *text* LLM is the same problem, unaddressed. In the grounded corpus
pass ([`render_grounded_mission_corpus.py`](../../../../source/strafer_lab/scripts/render_grounded_mission_corpus.py),
in-Kit), if `--use-planner-llm` / `--use-paraphrase-llm` are passed,
[`build_default_waypoint_runner`](../../../../source/strafer_lab/strafer_lab/tools/build_mission_queue.py)
(and the paraphrase runner that reuses it) loads `Qwen/Qwen3-4B` via
`from_pretrained(device_map="auto")` **inside the Kit process** → pulls
`accelerate` → flips `cuda-bindings` `12.9.4` → `13.0.3` → the identical
`torch 2.10.0+cu130` (pins `==13.0.3`) vs `isaacsim-core 6.0.0` (pins
`==12.9.4`) conflict the VL pivot removed.

It cannot be two-passed away: grounding interleaves with generation (a
same-room "no" re-rolls the start pose and re-plans), so the planner must
be callable *mid-traversal*. The measured grounded corpus therefore ran
the **oracle** planner + **templated** paraphrases (no LLM) — which is
why it ran clean. (The provenance — `generator_metadata["llm_model"]`
stamping the model only on real LLM rows — is already correct on `main`;
this brief does not touch it.)

### The design: extend serve-planner, swap to HTTP-client runners

`strafer_autonomy` **already runs an out-of-process Qwen3-4B service**:
`make serve-planner` → `uvicorn
strafer_autonomy.planner.app:create_app --factory --host 0.0.0.0 --port
8200`, in `vlm_env`. The model is loaded once at startup
([`LLMRuntime`](../../../../source/strafer_autonomy/strafer_autonomy/planner/llm_runtime.py)).
But its existing endpoints (`POST /plan`, `POST /plan_with_grounding`) do
command → `MissionIntent` → `MissionPlan` — **not** waypoint generation
or paraphrasing.

So "reuse serve-planner" means **extend that FastAPI app with the
text-generation endpoint(s) the mission generator needs** (it already has
`Qwen3-4B` loaded — reuse the model + process + env), then replace
`build_default_waypoint_runner` / `build_default_paraphrase_runner` in
`build_mission_queue.py` with **HTTP-client runners** that POST to those
endpoints. The mission generator (in Kit) calls `:8200` over localhost;
`Qwen3-4B` stays in `vlm_env`, out of Kit, so the canonical isaaclab env
keeps `cuda-bindings==12.9.4` untouched.

This mirrors how the grounding pass keeps models out of Kit — the same
out-of-process treatment, but over HTTP rather than by going model-free
(the text pass genuinely needs the LLM; geometry was enough for
grounding). Note that the closed out-of-process *VL grounding* service
proposal is **not** this: that one was made moot by the geometric pivot;
the text LLM has no geometric substitute, so the HTTP service is the
real path here.

### Proposed endpoint shapes

The in-process runners are the contract to preserve. They live behind
two protocol seams in `build_mission_queue.py`:

```
WaypointRunner  = Callable[[str, int], str]          # (prompt, seed) -> raw JSON string
ParaphraseRunner = Callable[[str, int, int], list[str]]  # (mission_text, n, seed) -> lines
```

`build_default_paraphrase_runner` already just *wraps*
`build_default_waypoint_runner` with a different prompt + a line-split.
So one generic text endpoint serves both.

**Option A (recommended): one generic `/generate` endpoint.**

- `POST /generate`
  - request: `{ "request_id": str, "prompt": str, "seed": int,
    "max_new_tokens": int = 512 }`
  - response: `{ "text": str }`
  - server: `manual_seed(seed)`, `messages = [{"role": "user",
    "content": prompt}]`, run `LLMRuntime.generate(messages, seed=seed,
    max_new_tokens=max_new_tokens)` (greedy, `do_sample=False`,
    `enable_thinking=False` — already the runtime default), return the
    raw decoded text. This is byte-for-byte what the in-process
    `build_default_waypoint_runner._run(prompt, seed)` returns.
- HTTP **waypoint** runner: POST the `build_waypoint_prompt(...)` string
  + `config.llm_seed + attempt`, `max_new_tokens=512`; return
  `resp["text"]` — `parse_waypoint_json` consumes it unchanged.
- HTTP **paraphrase** runner: build the *same* paraphrase prompt the
  in-process runner builds, POST `/generate`, split lines client-side.
  The paraphrase scaffolding stays client-side, exactly as it does
  in-process, so the call sites behind the runner seam are unchanged.

Recommended because the paraphrase runner already reuses the waypoint
runner — one generic endpoint matches the existing structure and keeps
prompt-building / parsing on the `strafer_lab` side where it already
lives.

**Option B (if the service team prefers task-specific endpoints):** two
endpoints — `POST /generate_waypoints {request_id, prompt, seed} ->
{raw}` and `POST /paraphrase {request_id, mission_text, n, seed} ->
{paraphrases: [str]}` — moving the paraphrase prompt-build + line-split
server-side. Equivalent: request bodies are the existing waypoint /
paraphrase inputs and responses are the same strings the in-process
runners return.

Either way, the runners return the same `str` / `list[str]` the
in-process runners did, so `build_mission_queue`'s call sites
(`_plan_one_mission`, `_emit_row`) are untouched behind the seam.

### Cross-lane boundary

This spans two lanes; the brief calls it out so the PR sequencing is
explicit.

- **`strafer_autonomy` lane (planner service owner) owns:** the new
  endpoint(s) in `planner/app.py`; the request/response payload models in
  `planner/payloads.py`; and extending `LLMRuntime.generate` to accept a
  **per-request `seed` and `max_new_tokens`** — today it uses a fixed
  `self._max_tokens` (default 256) and sets no seed, but the waypoint
  pass needs 512 tokens and a per-call seed (`config.llm_seed + attempt`,
  so retries differ). The service already loads `Qwen3-4B` in `vlm_env`,
  so there is **no new model, process, or env** — only new routes over
  the loaded model. Adding endpoints here is cross-lane work to
  coordinate with the autonomy lane; it should not land silently from the
  harness PR.
- **`strafer_lab` lane (this brief / DGX) owns:** the HTTP-client runners
  (`build_http_waypoint_runner` / `build_http_paraphrase_runner`)
  replacing the `from_pretrained` in-process runners in
  `build_mission_queue.py`, and the CLI wiring in `build_mission_corpus.py`
  (a `--planner-service-url`, default `http://localhost:8200`, consulted
  when `--use-planner-llm` / `--use-paraphrase-llm` are passed; the
  grounded sibling inherits it because it reuses the corpus parser). No
  Kit-side model load remains.

## Acceptance criteria

- [ ] **serve-planner extended (cross-lane).** `serve-planner` gains the
      generate endpoint(s) above; `LLMRuntime.generate` accepts a
      per-request `seed` + `max_new_tokens`. Request = the waypoint /
      paraphrase prompts; response = the same strings the in-process
      runners return. Health / readiness / error status codes match the
      existing `/plan` route (503 until the model is loaded).
- [ ] **HTTP-client runners.** `build_default_waypoint_runner` /
      `build_default_paraphrase_runner` are replaced (or shadowed) by
      HTTP-client runners with the **same** `(prompt, seed) -> str` /
      `(mission_text, n, seed) -> list[str]` signatures, so
      `build_mission_queue`'s call sites are unchanged. No
      `from_pretrained` / `transformers` / `accelerate` import remains on
      the generator (Kit) side.
- [ ] **CLI wiring.** `build_mission_corpus.py` gains
      `--planner-service-url` (default `http://localhost:8200`), consumed
      only when `--use-planner-llm` / `--use-paraphrase-llm` is passed;
      `render_grounded_mission_corpus.py` inherits it via the shared
      parser.
- [ ] **The conflict is gone.** A grounded run with `--use-planner-llm`
      imports no `transformers` / `accelerate` inside Kit; `import
      isaacsim` + `import isaaclab_tasks` still succeed and the canonical
      isaaclab env keeps `cuda-bindings==12.9.4` (no env mutation).
- [ ] **Determinism preserved.** Same `llm_seed` + prompt → same waypoint
      string (greedy `do_sample=False`, server honours `manual_seed`);
      the waypoint validation / retry loop, the `path_shape_unsatisfied`
      fallback, and the cache key are unchanged.
- [ ] **Graceful degradation.** Service unreachable → the runner surfaces
      the failure as the empty / exception path
      `_plan_one_mission` already handles, so the existing retry →
      oracle-fallback engages with no crash; passing `--use-planner-llm`
      with no service up yields a clear operator error.
- [ ] **Test.** A stubbed-service unit test (no GPU, no Kit) asserts the
      HTTP runner threads `prompt` + `seed` to the endpoint and returns
      the body string; the mission-generator round-trip suite stays green
      with the runner swapped for the stub.
- [ ] **Doc surface.** `serve-planner`'s new text-generation endpoint(s)
      documented in
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      (the route + the `vlm_env` / isaaclab-env split);
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      scripts inventory notes the HTTP runner; and the `mission-generator`
      brief's planner-pass note updated to point at it.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit.

## Investigation pointers

- In-process runners to replace:
  `build_default_waypoint_runner` / `build_default_paraphrase_runner` +
  the `WaypointRunner` / `ParaphraseRunner` protocol types in
  [`build_mission_queue.py`](../../../../source/strafer_lab/strafer_lab/tools/build_mission_queue.py).
- The runner-selection + provider-threading site:
  `run()` in
  [`build_mission_corpus.py`](../../../../source/strafer_lab/scripts/build_mission_corpus.py).
- serve-planner internals:
  `create_app` + the `/plan` route in
  [`planner/app.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/app.py);
  `LLMRuntime.generate` in
  `planner/llm_runtime.py` (extend for `seed` + `max_new_tokens`);
  the payload models in `planner/payloads.py`; the retry / backoff
  client pattern to mirror in
  `clients/planner_client.py`; the `serve-planner` Make target (`vlm_env`,
  port 8200).
- The `cuda-bindings` conflict this avoids is documented in the
  `mission-generator` brief's grounding section; the env split (Qwen in
  `vlm_env`, canonical isaaclab env at `12.9.4`) is the whole point.

## Out of scope

- **The geometric grounding pass.** Already out of Kit and model-free; no
  service needed for it.
- **Replacing the deployed `/plan` pipeline.** The command →
  `MissionIntent` → `MissionPlan` route stays as is; this brief only adds
  text-generation endpoint(s) over the same loaded model.
- **The grounder-finetune VL service** (`strafer_vlm` `/ground`).
  Different consumer, different model; untouched.
- **Multi-host / remote serving.** localhost colocation on the DGX is the
  target; LAN serving is the deployed planner's concern, not the
  offline corpus generator's.
- **LLM text-quality benchmarking.** Whether the LLM text beats the
  templated baseline is the `mission-generator` brief's hallucination
  benchmark; this brief only changes *where the model runs*, not whether
  it is worth running.
</content>
</invoke>
