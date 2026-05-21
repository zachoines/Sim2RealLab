# Calibrated confidence and uncertainty on `RoomEntry`

**Type:** new feature
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (small additive surface; unblocks
uncertainty-driven re-visitation in the parked
[`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)
extension)
**Estimate:** S (~1–2 days; temperature scaling + entropy
field + calibration measurement against the eval harness)
**Branch:** task/room-state-uncertainty-calibration

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
merges. Recommended ordering: file alongside
[`room-state-eval-harness`](room-state-eval-harness.md) so
calibration can be measured (ECE requires the harness).

## Story

As a **planner deciding whether to commit to
`room_anchor(label)` as a cross-room transit destination or
fall back to exploration**, I want **`RoomEntry.confidence` to
be a calibrated probability and `RoomEntry.uncertainty` to
expose epistemic uncertainty**, so that **low-confidence
cluster labels trigger active disambiguation rather than
over-confident wrong-room commitments, and the autonomy-stack
consumer can act on a single scalar signal that's actually
meaningful**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — where this brief sits in the v1 / v1.5 / v2 / v2.5 / v3 / escape-valve stack, and which planner-side consumers depend on its output.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — v1 implementation. This brief modifies its
  `RoomClassifier` and `RoomEntry`.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  ECE measurement plumbed there; this brief activates it.
- [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)
  — parked consumer. The v1.5 "re-visitation gain" section
  (this brief's twin) acts on `RoomEntry.uncertainty`.

## Context

### Why the current "confidence" is wrong

v1's `RoomEntry.confidence` is the mean cosine similarity
between the cluster's per-node CLIP image embeddings and the
top-1 text prompt embedding. Cosine similarity is not a
probability:

- Its range under L2-normalization is `[-1, 1]`, with typical
  CLIP zero-shot values in `[0.15, 0.30]` even for confidently
  correct labels.
- It carries no calibration — the network has no incentive to
  assign meaningful confidence levels during contrastive
  training.
- There's no epistemic uncertainty signal: a node that has
  cosine 0.25 to "kitchen" and 0.24 to "living room" carries
  the same `confidence` as one that has 0.25 to "kitchen" and
  0.10 to everything else, even though the first is much more
  ambiguous.

Downstream consumers either ignore the field or use it as a
threshold without justification. The autonomy-stack compiler
in [`autonomy-stack`](autonomy-stack.md) reads
`known_rooms[X].observed_objects` for target-room inference
but does not gate on `confidence`, partly because the field
isn't meaningful enough to gate on.

### Two fixes, paired

**(a) Temperature-scaled softmax for calibration.**
Compute per-prompt cosine similarities, divide by a learned
temperature `T`, apply softmax. `T` is fit once offline
against the
[`room-state-eval-harness`](room-state-eval-harness.md) eval
set to minimize ECE (Expected Calibration Error). Output is a
proper probability vector over the prompt set; `confidence`
becomes the top-1 probability.

**Initialize `T` from CLIP's `logit_scale`, not from `1.0`.**
OpenCLIP ViT-B/32 ships a learned scalar `logit_scale ≈ 1/0.07
≈ 14.3` (clipped to a max of 100, i.e., `T_min ≈ 0.01`) that
the model was trained to use when converting cosine
similarities to logits. Raw cosine similarities for room
prompts on the existing backbone sit in `[0.15, 0.30]`;
softmax over that range with `T = 1.0` produces an almost
uniform distribution and ECE is uninformative. Start the
ECE search from `T_init = 0.07` (equivalently, multiply
sims by `1/T = 14.3`), bracket the search in
`T ∈ [0.01, 0.5]`, and report whether the optimum is
materially different from the CLIP-trained value — if not,
the calibration step is a one-line constant and the
follow-ups (deep ensemble, MC-dropout) can be retired
faster. Source for the `logit_scale` value:
[OpenCLIP](https://github.com/mlfoundations/open_clip/discussions/763);
the clip-to-100 detail is documented in
[openai/CLIP#46](https://github.com/openai/CLIP/issues/46).

**(b) Entropy-based uncertainty field.**
Expose `RoomEntry.uncertainty: float` populated from the
entropy of the cluster's aggregated label distribution
(softmax mean over member nodes). Low entropy = label is
unambiguous; high entropy = label is contested. Normalized to
`[0, 1]` by dividing by `log(N_prompts)`.

The two fields are complementary: `confidence` is "how
certain is the model that this is room X"; `uncertainty` is
"how uncertain is the model overall about this cluster's
label". A high-confidence-high-uncertainty pair flags the
case where the cluster has many nodes with very different
labels — the v1 cluster averaging masks this today.

### Algorithm sketch

```python
# In RoomClassifier:
def __init__(self, ..., temperature: float = 1.0):
    self._temperature = temperature

def classify_distribution(
    self, image_embedding: np.ndarray,
) -> np.ndarray:
    """Return calibrated probability vector over the prompt set."""
    sims = self._text_embeddings @ image_embedding
    scaled = sims / self._temperature
    return softmax(scaled)

# In aggregate_room_entries / cluster-level:
def aggregate_label_distribution(
    graph, member_ids,
) -> np.ndarray:
    """Mean of per-node label distributions across cluster."""
    dists = [graph.nodes[nid]["data"]["metadata"]
                 .get("room_label_dist") for nid in member_ids]
    return np.mean([d for d in dists if d is not None], axis=0)

# RoomEntry gains:
#   confidence: top-1 probability of the cluster's aggregated dist
#   uncertainty: H(dist) / log(N_prompts)
```

Temperature `T` is fit via the eval harness:

```python
T_star = argmin_{T > 0} ECE(eval_set, classifier(T))
```

The fit script ships in
`source/strafer_lab/scripts/fit_room_classifier_temperature.py`
and writes the temperature to
`source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py`
as the new module-level default (or a JSON sidecar the
classifier reads at construction).

### Tie-in to state-of-the-art

- **Temperature scaling**: Guo et al. 2017 (arXiv:1706.04599)
  — the canonical calibration method for neural classifiers.
  One scalar, post-hoc, no retraining.
- **Predictive entropy as epistemic uncertainty**: Gal &
  Ghahramani 2016; standard signal in active-perception
  literature.
- **ECE as the calibration metric**: same Guo et al. paper.

The temperature-scaling choice is deliberately the simplest
calibration primitive — deep ensembles or MC-dropout require
running multiple forward passes per observation, which the
manager's per-observation latency budget can't absorb.

## Acceptance criteria

- [ ] **Temperature scaling in `RoomClassifier`.** Constructor
      accepts a `temperature: float = 0.07` argument
      (initialized to CLIP's trained `logit_scale ≈ 1/0.07`;
      see "Initialize `T` from CLIP's `logit_scale`" above).
      The `classify` method's confidence output is the
      top-1 probability of the temperature-scaled softmax,
      not the raw cosine similarity.
- [ ] **`RoomEntry.uncertainty` field.** A new float field
      on
      [`RoomEntry`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py),
      populated from the entropy of the cluster's mean
      label distribution. Normalized to `[0, 1]`.
- [ ] **Per-node distribution stamping.**
      `add_observation` stamps
      `metadata["room_label_dist"]` (the full softmax vector)
      alongside the existing `room_label` / `room_conf`, so
      cluster aggregation can compute the mean distribution
      without re-encoding.
- [ ] **Temperature-fit script.**
      `source/strafer_lab/scripts/fit_room_classifier_temperature.py`
      runs the eval harness over a range of `T` values and
      writes the best-ECE temperature to a JSON sidecar at
      `source/strafer_autonomy/strafer_autonomy/semantic_map/room_classifier_calibration.json`.
      Default `T=1.0` (no calibration) ships if the script
      hasn't been run yet.
- [ ] **ECE measurement.** PR description carries the
      uncalibrated vs. calibrated ECE on the eval harness's
      four scenes. Calibrated ECE ≤ 0.1 on at least three of
      four scenes.
- [ ] **Backward-compatible.** `RoomEntry.confidence` keeps
      its existing field name and semantics-shape (a `[0, 1]`
      scalar where higher is better). Consumers reading the
      field get a better-calibrated number, not a different
      one.
- [ ] **Unit tests.** Synthetic prompt-set / image-embedding
      fixtures verifying: softmax produces a valid
      distribution; entropy field is computed correctly;
      `T=∞` produces uniform distribution; `T→0+` produces
      one-hot.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- Existing classifier:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `RoomClassifier.classify`.
- Existing `RoomEntry`:
  [`semantic_map/models.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py).
- Calibration reference: Guo et al. 2017
  (arXiv:1706.04599).
- ECE implementation: sklearn has `calibration_curve` but
  not a direct ECE helper; a 20-line bin-based ECE
  computation ships with the eval harness.

## Out of scope

- **Deep-ensemble or MC-dropout uncertainty.** Heavier than
  the manager's per-observation latency budget can absorb.
  Filed as a v3 alternative if temperature scaling proves
  insufficient.
- **Per-prompt temperature.** Single scalar `T` for the
  whole prompt set. Per-prompt would over-fit the small eval
  set.
- **Active disambiguation logic in the planner.** This brief
  exposes the signal; the consumer behavior (re-visit
  uncertain rooms) lives in the parked
  [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)
  brief's v1.5 extension.
- **VLM-derived uncertainty.** The VLM-refined label work in
  [`room-label-vlm-refinement`](room-label-vlm-refinement.md)
  may surface its own confidence; combining VLM and CLIP
  uncertainties is a follow-up.
