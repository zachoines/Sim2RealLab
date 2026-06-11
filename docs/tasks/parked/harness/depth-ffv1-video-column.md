# Migrate depth from per-frame PNG sidecar to FFV1 lossless 16-bit video

**Type:** harness storage-format change (depth representation; replaces the per-frame PNG sidecar)
**Owner:** DGX agent
**Priority:** P2 — does not block any acceptance bar (the PNG sidecar works today and is round-trip-tested), but it is an operator-committed improvement: it removes the "many small files" read-throughput cost at Tier-2/3 scale **and** buys loader ergonomics (depth becomes a first-class video feature instead of a path-convention sidecar). The accepted trade is taking on the experimental-feature-API + LeRobot-codec coupling in exchange for both wins.
**Estimate:** M (~2–3 days; can balloon if LeRobot's video-encode path resists per-feature codec override — see Risks). Structured spike-first so the uncertain part is proven before the integration work.
**Branch:** `task/depth-ffv1-video-column`

**Blocked on / sequencing:**
- **Sequenced *after* the R1 first-class `observation.detections.*` column lands.** Both changes edit `lerobot_writer.py` / `build_features`. R1 is critical-path gated work ([`vlm-grounding-finetune`](../clip-validation/vlm-grounding-finetune.md) + the validator's case-2 alternates depend on it). Land R1 first, then this, to avoid writer churn / merge collisions. Two sequential PRs, not one.
- Original gate on [`teleop-perf-architecture`](../../completed/teleop-perf-architecture.md) (no concurrent capture-loop edits) is **satisfied** — it shipped 2026-06-01 (PR #73).
- **No data migration.** Per the harness clean-break stance, no real LeRobot corpus is captured yet. This is a format swap, not a data migration — there are no existing PNG datasets to convert. (If the Tier-1 acceptance corpus lands before this brief is picked up, that corpus is a re-capturable acceptance artifact, not training data — still no migration tooling.)

## Story

As a **harness operator capturing and reading depth at Tier-2/3 scale** I want **depth stored as one lossless 16-bit FFV1 video stream per shard instead of thousands of per-frame PNGs, registered as a first-class LeRobot feature** so that **read throughput doesn't degrade on the small-files problem, and a stock `LeRobotDataset` consumer gets `observation.depth.perception` through the normal loader instead of having to know our sidecar path convention.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`active/harness/harness-architecture.md`](../../active/harness/harness-architecture.md) — the spec this revises. The [`Depth representation`](../../active/harness/harness-architecture.md#depth-representation--sidecar-png-sequence) section records *why* PNG-sidecar was chosen over the `StraferDepthSequenceFeature` registered-feature route (experimental `register_feature` API; 16UC1 fits no native dtype). This brief reverses that, accepting the coupling for the small-files + ergonomics wins. The dir-tree diagram, the features table row, and the Tier-1 implementation checklist item all reference the depth layout and must move together.
- [`strafer_lab.tools.lerobot_depth`](../../../../source/strafer_lab/strafer_lab/tools/lerobot_depth.py) — the current PNG helpers. The meters↔mm contract (`_DEPTH_SCALE_M_PER_UNIT = 0.001`, invalid-depth → 0, clip to `[0, 65535]` mm) is **preserved**; only the on-disk container changes. `test_lerobot_depth.py` is the existing round-trip oracle.
- [`depth_downsampler.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py) — the real-robot 16UC1-millimeters convention. The FFV1 decode must yield the same meters as the real stack and as `read_depth_png`, so sim-real format match is unchanged.

## Motivation

Depth cannot ride LeRobot v3's native `video` feature: that path is MP4/H.264, which is 8-bit per channel and quantizes away the 16-bit (1 mm) precision depth needs. The shipped workaround is a per-frame 16UC1 PNG sidecar — lossless, zero-coupling, but it produces **one tiny file per frame** (a 600-frame episode = 600 PNGs; Tier-3's thousands of episodes = millions of files), which hits the filesystem-metadata / no-sequential-streaming "small-files" read penalty.

**FFV1** is a lossless intra codec that supports **`gray16le` (16-bit single-channel)** bit-exactly. One FFV1 stream per shard collapses the file count by ~3 orders of magnitude *and* — if registered as a LeRobot video feature — gives stock consumers depth through the normal loader. The cost (accepted by the operator): FFV1 is not a valid MP4 stream (it needs an MKV/NUT container, diverging from the RGB `.mp4` streams), and LeRobot's video-encode path has hardcoded codec assumptions (we already hit `vcodec="libx264"`→`"h264"` and the `libsvtav1` exclusion on ARM64), so a per-feature codec override or a thin subclass is required — the "experimental-API + monkey-patching" coupling the PNG route was chosen to avoid.

The two wins (small-files + ergonomics) are judged worth that coupling. This brief de-risks the load-bearing uncertainty (does FFV1 `gray16le` round-trip bit-exactly through our **PyAV** decode path on ARM64, where torchcodec is excluded?) with a spike *before* the LeRobot integration, and names a zero-coupling fallback if the spike fails.

## Phase 1 — spike (gate the rest on this; ~0.5 day)

Prove the lossless round-trip on the **actual** ffmpeg/PyAV builds, standalone, with no LeRobot involvement:

- [ ] A standalone PyAV script encodes a known `(T, H, W)` uint16 array to **FFV1 / `gray16le`** (MKV container) and decodes it back, asserting `np.array_equal(decoded, original)` — **bit-exact**, not approximate. Cover edge values (0 = invalid sentinel, 65535 = clip ceiling, a mid-range gradient).
- [ ] Run it on **both** the DGX `.venv_harness` PyAV **and** a Jetson-side PyAV (the Jetson reads depth for training/eval; torchcodec is excluded there → PyAV is the decode path). Confirm both ffmpeg builds expose the FFV1 encoder + `gray16le` pixel format.
- [ ] **Pixel-format trap:** verify no implicit colorspace/limited-range conversion sneaks in (PyAV can route through RGB/limited-range YUV). The stream must be `gray16le` end-to-end. A single non-bit-exact pixel fails the spike.

**Decision gate:** if the spike is bit-exact on both stacks → proceed to Phase 2. If FFV1/`gray16le` is unavailable or not bit-exact on ARM64 → **stop and fall back** (see Fallback); do not sink time into the LeRobot integration.

## Phase 2 — LeRobot integration + writer/reader

- [ ] Find LeRobot v3's video-encode entry point (the `encode_video_frames`-equivalent the writer calls from `save_episode`/`finalize`) and the **minimal** way to make depth use FFV1/`gray16le`: prefer a **per-feature codec config** if v3 exposes one; else a thin **subclass** of the writer's encode step; monkey-patching the module function is the last resort. Pin to the installed LeRobot version (`0.5.1`) and add a guard test that fails loudly if the upstream encode signature changes.
- [ ] Register `observation.depth.perception` (and `observation.depth.policy` when the policy cam is on) as a **first-class video feature** in `build_features`, encoded FFV1/`gray16le`, container `.mkv`. Confirm LeRobot's `meta/info.json` tolerates a per-feature non-`.mp4` path/codec alongside the RGB `.mp4` streams (mixed containers in `videos/`) — this is a spike sub-item of Phase 2; if the reader hard-assumes `.mp4`, that override moves here too.
- [ ] Rework [`lerobot_depth.py`](../../../../source/strafer_lab/strafer_lab/tools/lerobot_depth.py): replace the per-frame `frame_path` / `episode_dir` PNG API with FFV1 encode/decode helpers (accumulate frames per episode → encode on `save_episode`; decode a stream → `(T, H, W)` float32 meters). **Preserve the meters↔mm contract exactly** (`* 1000` on write, `* 0.001` on read, invalid→0, clip `[0, 65535]` mm) so consumers see no semantic change — only the container changes. Keep the PNG helpers available behind the fallback path until the migration is proven, then retire them.
- [ ] Extend `test_lerobot_depth.py`: the FFV1 round-trip must be **bit-exact against the same arrays the PNG helpers round-trip** (`read(ffv1(x)) == read_depth_png(write_depth_png(x))`). The PNG format stays the reference oracle for the migration. `make test-harness` stays green.

## Phase 3 — spec update (same PR as Phase 2)

- [ ] Rewrite [`harness-architecture.md`](../../active/harness/harness-architecture.md)'s [`Depth representation`](../../active/harness/harness-architecture.md#depth-representation--sidecar-png-sequence) section from "sidecar PNG sequence" to "FFV1 lossless 16-bit video feature," recording the reversal and the accepted trade.
- [ ] Update the three coupled references in that doc that name the PNG layout: the dataset dir-tree diagram (the `observation.depth.perception/episode-NNNNNN/NNNNNN.png` block), the per-frame features table row (`observation.depth.perception` → first-class video feature, FFV1/`gray16le`, `.mkv`), and the Tier-1 implementation checklist item that currently reads "Implement `StraferDepthSequenceFeature`… 16UC1 PNG."

## Acceptance

- [ ] Phase 1 spike bit-exact on DGX **and** Jetson PyAV stacks (or fallback taken, with the spike result recorded in the brief before it ships).
- [ ] Depth registered as a first-class LeRobot video feature; a stock `LeRobotDataset` round-trip loads `observation.depth.perception` through the normal loader and decodes 16UC1 → float32 meters bit-exactly.
- [ ] Per-episode/per-shard file count for depth drops from O(frames) to O(shards) — the small-files win is demonstrated (count files under `videos/observation.depth.*` before/after for a fixed episode set).
- [ ] `meters↔mm` semantics and sim-real format match ([`depth_downsampler.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py) 16UC1 mm) unchanged; FFV1 decode == PNG decode for the same arrays.
- [ ] LeRobot-version guard test present; `make test-harness` green.
- [ ] Brief shipped to [`completed/`](../../completed/) per [`conventions.md`](../../context/conventions.md) inside the shipping PR (stamp with the work commit + PR link).

## Fallback (if Phase 1 fails)

If FFV1/`gray16le` does not round-trip bit-exactly through PyAV on ARM64, do **not** force it. Fall back to a **per-episode-packed sidecar**: one file per episode — a `(T, H, W)` uint16 `.npy`/`.npz` (or zarr / multi-page 16-bit TIFF) — instead of one PNG per frame. This still collapses the small-files count by ~3 orders of magnitude and keeps the PNG route's zero-LeRobot-coupling + "format-is-the-contract" properties; it just doesn't buy the first-class-loader ergonomics. Record the spike result and re-file this brief's scope as the packed-sidecar variant.

## Risks

1. **PyAV/ARM64 FFV1 support** — the load-bearing uncertainty; Phase 1 gates on it.
2. **LeRobot encode-path coupling** — the override may be brittle across upstream bumps; pin the version and guard-test the signature. If the only path is a deep monkey-patch, weigh against the packed-sidecar fallback (which has none of this risk).
3. **Mixed containers in `videos/`** — RGB `.mp4` + depth `.mkv` in the same tree; verify the LeRobot reader indexes per-feature container/codec rather than assuming `.mp4`.
