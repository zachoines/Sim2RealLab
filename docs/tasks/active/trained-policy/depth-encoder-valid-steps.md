# Encode only the valid steps of a recurrent mini-batch with the frozen DeFM encoder

**Type:** bug
**Owner:** DGX
**Priority:** P1. Every depth-policy training run steps its host memory by 14–16 GiB at a time during the policy
update, and v4's 1000-iteration run is next.
**Estimate:** S (one method and its tests), plus one GPU run to read the memory and the learning time
**Branch:** `task/depth-encoder-valid-steps`

## Story

As **the depth-policy training run**, I want **the policy update to push only real rollout steps through the frozen
depth encoder**, so that **its memory no longer grows with the number of episode ends in a rollout and the update
does the same arithmetic on every valid step (to convolution rounding on the GPU).**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)

## Context

- **The padding.** rsl_rl's recurrent update (`RolloutStorage.recurrent_mini_batch_generator`) splits each
  environment's 48-step rollout at its episode ends and pads every trajectory to 48 steps. A mini-batch of 24
  environments is `(48, n, D)` with `n` = 24 plus the episode ends among them before the rollout's last step (an end
  on the last step adds no trajectory); `masks` marks the valid steps.
- **The cost.** `StraferDepthRNNModel.get_latent` flattened all `48 n` rows, padding included, through the DeFM
  encoder, for the actor and for the critic, in every mini-batch and epoch. Its largest activation (the first
  expand stage, 96×112×112 float32) is 220.5 MiB per padded trajectory, and two such tensors are live at once.
- **What it does to memory.** A mini-batch with more trajectories than any since the cache last emptied takes a new
  pair of blocks from the driver; PyTorch's caching allocator keeps the old pair. With `cudnn.benchmark` on (Isaac
  Lab's seeding sets it), each first-seen `n` is also a new convolution shape, so cuDNN searches and then empties
  the cache. Two 15-iteration instrumented resumes of v3's training (2026-11-15) read it directly:
  - the allocator's reserved figure moved one-for-one with the host's MemAvailable;
  - it rose only at a new largest `n`, to 50.7 GiB, and fell only at a first-seen `n`;
  - updates with a first-seen `n` peaked at 22.4–31.0 GiB allocated, those without at 15.7–17.7 GiB;
  - updates without a search took 0.511–0.518 s of learning per padded trajectory.
- **Why only the valid steps can be dropped.** Padding follows each trajectory's last valid step. The GRU's output
  at a valid step depends only on earlier steps, and `RNN.forward` keeps only the valid steps
  (`unpad_trajectories`) before the heads and the loss. The DeFM preprocessing, backbone and projection each act on
  one image at a time. So the valid rows' embeddings do not depend on the padded rows, up to convolution rounding:
  kernels are chosen per batch size, so on the GPU a row's embedding can move in its last digits with the batch.
- **What the RNN sees at padded steps.** They enter the normalizer as zeros, so the RNN gets −mean/(std + eps) there
  instead of the normalized encoding of an all-zero image. Either way they are dropped before the heads.
- **The CNN fallback encoder is different.** Its BatchNorm normalizes over its batch in training, so removing the
  padded rows would change its outputs. It keeps the padded batch.

## Acceptance criteria

- [ ] In the recurrent update (`masks` given) the DeFM encoder sees `masks.sum()` rows, `48 × envs per mini-batch`
      whatever `n` is; padded steps enter the normalizer as exact zeros.
- [ ] Pure tests, CPU only, against a padded-batch reference (every padded row encoded) on padded rollouts built by
      rsl_rl's own `act` and `process_env_step`, at every rollout seed from 0 to 59:
  - in every mini-batch, the valid rows' embeddings, the latent, the actor's distribution parameters,
    log-probabilities and entropy, and the critic's values equal the reference's;
  - one `PPO.update` step from identical parameters, on each mini-batch: the losses and every gradient equal the
    reference's, except the depth projection's. Those are sums over the encoder's rows, which the reference takes
    with extra exact zeros, so they may differ in the order of a float32 sum. The test shows the summed terms are
    identical and the zeros exact, and bounds the difference by the summation error. The gradient-norm clip and
    Adam can carry it into any parameter of the same model, so the parameters after the step are held to
    4 eps (|p| + lr) of the reference's;
  - values put at the padded steps leave a whole update exactly as with zeros there, on both paths;
  - without masks, and with the CNN encoder whether requested or DeFM's fallback, every row is encoded as in the
    reference.
- [ ] One GPU run on the same launch as the instrumented resume above, beside that resume run as its baseline, with
      the clock cap in force. From the second update on: reserved memory holds its level (no rise of 1 GiB or more),
      allocated peaks at 12.0–13.5 GiB in every update (predicted 12.6–12.9 GiB), and no release inside learning.
      Learning time is predicted at about 49 s per iteration and reported against the baseline's.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or
      guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-lab-pure` passes.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`: `get_latent`,
  `_encode_depth`, `_encode_valid_steps`.
- `source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_encoders.py`: `DeFMDepthEncoder.forward`.
- rsl_rl 5.4.2: `storage/rollout_storage.py` (`recurrent_mini_batch_generator`), `utils/utils.py`
  (`split_and_pad_trajectories`, `unpad_trajectories`), `modules/rnn.py` (`RNN.forward`), `algorithms/ppo.py`
  (`update`).

## Out of scope

- **The frozen backbone's training mode.** rsl_rl's `train_mode()` reaches the DeFM backbone, so its nine
  stochastic-depth blocks drop residual branches at random per row in every training forward pass; deployment runs
  it in eval mode. On the CPU, with the cached weights and synthetic room-like depth, the 128-d embedding moved by
  15–18 % (relative L2) against eval mode, and two draws differed by 23 %. This change keeps that behaviour; the
  valid rows get draws from the same distribution, not the same draws. Pinning the backbone to eval mode changes
  training and is a decision of its own. Two facts go with it: `copy.deepcopy` of the real backbone returns it in
  train mode whatever the source's mode (the export paths call `.eval()` afterwards), and the hub reference
  `leggedrobotics/defm:main` that `_load_defm_backbone` loads is not pinned.
- Encoding each rollout step once and reusing it across epochs and mini-batches.
- Any allocator setting (`empty_cache`, a memory fraction, `expandable_segments`).
