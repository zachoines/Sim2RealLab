# Warm-start the GRU actor by behavior cloning on teleop demos, then fine-tune with stock PPO

**Type:** task
**Owner:** DGX agent
**Priority:** P2
**Estimate:** L — three separable pieces (a capture-side obs column, a BC trainer, a checkpoint bridge), each small; the cost is the demo campaign and the fine-tune legs.
**Branch:** task/bc-warm-start

## Story

As **the DGX agent**, I want **the recurrent actor initialized from human
demonstrations before PPO starts**, so that **goal-directed navigation does
not have to be discovered from scratch by exploration, and demo leverage
returns without a custom training loop.**

## Context bundle

Read these before starting:

- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- Related: [`harness-architecture`](../../active/harness/harness-architecture.md)
  (the capture matrix this lane draws from) and
  [`isaac-lab-upgrade`](../../active/tooling/isaac-lab-upgrade.md) (supplies the
  `torchcodec` prerequisite below).

## Context

The repo previously carried demo leverage as DAPG and GAIL auxiliary losses
bolted onto a monkey-patched copy of rsl_rl's `PPO.update()`. That machinery
is deleted: a hand-copied upstream training loop needs a re-merge on every
rsl-rl bump, and the losses were never the reason to want demos. Behavior
cloning gets the same leverage with **stock PPO** downstream — no custom
update loop anywhere — which is the constraint this brief exists to hold.

**Data source.** Demos come from the harness's **teleop driver's LeRobot
output** (`capture.py --driver teleop`), not from `collect_demos.py`'s HDF5.
The `(teleop, scene-metadata)` cell is wired today; `(teleop, queue)` is
pending the mission generator. **Coverage-mode capture is the wrong data** and
must not be substituted: it is deliberately diverse-perspective — the same
place from spread approach headings — which is the training signal for room
state and VPR, not for goal-directed imitation. That is a standing finding,
not a preference.

**The load-bearing gap is on the capture side, not the training side.** The
LeRobot writer's `observation.state` is a 10-dim pose + achieved-velocity
vector (`lerobot_writer.py:349-358`), and its `action` column is the
`[vx_cmd, vy_cmd, omega_z_cmd]` velocity command. Neither is what the policy
consumes or emits:

- the policy observation is the concatenated tensor in the **trained layout**
  — 19 dims for NOCAM, 3619 for DEPTH (`policy_interface.py:253-254`), whose
  term order and scales are pinned by the layout goldens in
  `test_composition_contract.py` and by `test_obs_contract.py`;
- the policy **action** is bounded `[-1, 1]` from `AffineBetaDistribution`,
  scaled to body velocities by `_velocity_scale` inside the action term
  (`mdp/actions.py:166-169, 317`). Recovering it by dividing the recorded m/s
  back out is lossy once the clamp saturates.

So the capture path needs to record the policy obs vector and the pre-scale
policy action as first-class columns. A demo set that does not carry the
trained layout is unusable for BC against that checkpoint, and the failure is
silent — shapes can agree while term order does not.

**Prerequisite (satisfied, but hard).** Reading LeRobot video back requires
`torchcodec==0.16.0` on the post-migration stack: torchvision 0.26 deleted the
`torchvision.io` video surface lerobot's default decode path routes to, and
`torchvision<0.26` is not pinnable under isaacsim-core 6.0.1.0. The
replacement is proven — value-level round-trip with depth bit-exact — but this
lane cannot read its own demos back without it.

## Acceptance criteria

- [ ] The teleop capture path records the policy observation vector **in the
      trained layout** plus the policy action, at policy cadence, as declared
      LeRobot columns. A test asserts the recorded obs column matches the
      layout the policy interface exposes for that variant — field order and
      scale, not just width.
- [ ] `ExpertDemoBuffer` reads `LeRobotDataset` instead of HDF5, preserving
      multi-episode concatenation into one uniformly-sampled transition pool.
      Its return-percentile episode filter is **not** ported as-is: the HDF5
      version appends returns only for episodes that carry `rewards` while
      appending obs/actions for every episode, so the keep-indices are computed
      over one list and applied to a longer one and a mixed corpus silently
      keeps the wrong episodes (`demo_buffer.py:87-101`). Either drop the
      filter or hold each episode's obs, actions, and return in one aligned
      record.
- [ ] The **BC entry point** derives the expected observation dimension and
      layout identity from the live policy variant and passes them to the
      buffer, and a mismatch is a hard error. Today's `expected_obs_dim`
      parameter has no callers, so "the buffer supports a check" is not the
      criterion — the caller supplying the live dimension is. Without it a
      19-dim NOCAM demo set trains against a 3619-dim depth policy silently.
- [ ] Sequence-aware BC trains the GRU actor with truncated BPTT over demo
      trajectories. Single-step BC is wrong for a recurrent policy and is not
      an acceptable fallback; the truncation window and hidden-state carry are
      stated and tested on a synthetic trajectory.
- [ ] A BC-trained actor plus a fresh critic is written as an
      **rsl_rl-compatible checkpoint** that `OnPolicyRunner` resumes without
      modification, verified by loading it back and matching state-dict keys
      and shapes against a stock-initialized runner.
- [ ] Fine-tuning runs on **stock PPO**: the training objective is upstream's
      unmodified `update()` — no copied loop, no subclass, no auxiliary-loss
      registry. (The existing `--lr_schedule` wrapper is a learning-rate
      decorator, not an objective change, and is out of scope here.)
- [ ] The critic-destruction hazard is addressed explicitly (critic warm-up
      and/or a conservative initial LR), with the chosen mitigation recorded
      alongside a measurement showing the BC initialization survives the first
      updates.
- [ ] `torchcodec==0.16.0` is recorded as a hard dependency of this lane
      wherever the env pins live.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — the pure
      strafer_lab suite stays green, and a short stock-PPO training smoke runs
      from a BC checkpoint.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tools/lerobot_writer.py:324-381` — the
  features dict; new columns are declared here.
- `source/strafer_lab/scripts/teleop_capture.py` — the in-process teleop
  driver that would populate them.
- `source/strafer_lab/strafer_lab/tasks/navigation/agents/demo_buffer.py` —
  `ExpertDemoBuffer`, the reader to refit.
- `source/strafer_shared/strafer_shared/policy_interface.py:155-254` — the obs
  field tables the recorded layout must match.
- `source/strafer_lab/test_sim/env/test_composition_contract.py:245-266` — the
  layout golden and the hash that pins it.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py:166-169,317`
  — the action scaling BC has to invert or bypass.
- `source/strafer_lab/scripts/capture.py:51-59` — which `(driver,
  mission-source)` cells are wired.

## Follow-on work this brief owns

- **Retire the HDF5 demo path.** Once the LeRobot path is proven,
  `scripts/collect_demos.py` and the HDF5 half of `ExpertDemoBuffer` become
  removable — one capture interchange format instead of two. This brief owns
  that deletion; it is not a separate filing.
- **Decide the fate of `encoded_obs_dim` / `encode_obs`**
  (`agents/depth_rnn_model.py`). The GAIL discriminator was their only caller,
  so they are dead code today, but a BC trainer may want exactly this encoder
  entry point to run demos through the depth compression. Keep or delete on
  the evidence of whether BC uses them.

## Out of scope

- Any change to the PPO update path. Stock rsl_rl, always — reintroducing a
  patched loop would recreate the maintenance problem the deletion solved.
- The mission generator that would wire `(teleop, queue)`. This lane runs on
  `(teleop, scene-metadata)` until that ships.
- Coverage-mode capture and its consumers.
- LeRobot's own trainers as a high-level behavior lane — a separate,
  unfiled exploration, not this brief.
- Starting the work. The training lane is paused behind the rig gate and the
  Isaac Lab migration; this brief is filed and queued, and un-parks when both
  clear.
