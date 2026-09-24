# The command debug markers: on the pre-flip stack, and in v3's loop — 2026-09-23

`debug-marker-leak-2026-09-21` left two questions open (§8 there):

- **Check 2:** did the stack every pre-flip depth policy trained on render the command markers
  into the D555?
- **Check 4:** does v3 behave differently in closed loop when the markers are positioned in
  every environment?

Both are answered here, each against a reading written down before it ran (`check2/`, `check4/`,
`PREREGISTRATION.md` in each).

- **The pre-flip stack drew the markers into the D555's RGB, not its depth.** With the markers
  on, the policy depth, its 3 600-cell grid and the perception camera's depth are bit-identical to
  the markers-off frame, while a real sphere at the same point changes all three. v1, v2, v2.1,
  v2.1a and cprime trained on the 80×45 policy camera measured here, with their markers
  positioned, and never had one in their depth input. v0 and the warm-start legs trained on the
  earlier 80×60 camera; for them that is an inference from the same marker code. Depth began to
  carry the markers only on the current pair, Isaac Sim 6.0.1.0 and Isaac Lab v3.0.0-beta2.patch1.
- **v3's steering changes with the markers positioned; no outcome metric moves beyond its
  standard error.** Completion is 0.888 against 0.905 (−0.017, standard error 0.021). The
  direction-offset median moves from 1.29° to 4.96° (+3.66°, standard error 0.34); the offset is
  the commanded direction's angle from the observed subgoal, positive counter-clockwise (left),
  as `eval_cadence_emulation.py`'s `signed_direction_offset` defines it — the attribution records
  measure against the mission goal instead (`debug-marker-leak-2026-09-21` §7). The share of left
  commands goes from 0.53 to 0.61, and the offsets spread less. That falls outside the
  indifference band written down before the runs, on the steering criterion only, and is
  recorded as v3's response to a novel object at its subgoal, not as a defect. Retraining for the
  markers is closed by construction: since #224 no camera can render a command marker.
- **v2, run the same way, is badly hurt by the markers** (completion 0.093 → 0.010), but from a
  baseline that has already collapsed on this tree. It shows the comparison detects a marker
  effect; it says nothing about what v2 saw in training.

## 1. Check 2 — did the pre-flip stack render the markers into the D555?

**Setup.** Tree `66c01a1` (the first parent of the #218 merge, the last pre-flip tree) under
Isaac Sim 6.0.0.0 and `IsaacLab-retired`. Before the boot, the bare interpreter showed `isaaclab`
resolving into `IsaacLab-retired` and `strafer_lab` into the worktree
(`check2/run/preboot_resolution.txt`). The launch matched pre-flip training: headless, cameras
on, no visualizer. On that Isaac Lab a command term's markers follow Kit's post-update event,
which every camera read pumps, so they are positioned without one.

**Method** (`check2/d555_marker_visibility_preflip.py`). v2's task cfg at one environment, with
the bridge's goal term and the 640×360 perception camera added. Both command families are placed
1 m ahead of the lens, and each is toggled on, off, on, off. Every state is one physics state
re-rendered in place, compared with the markers-off frame. Depth is judged by equality, RGB by
the markers' own colours. A real sphere (r 0.12) at the same point is the positive control. The
probe's JSON also carries a `projection` block computed from a stale camera pose; it is not
evidence of where the point sits — the frames and the positive control are.

**The markers were positioned.** After `env.step` alone, with nothing else called, the subgoal
sphere sat 0.0001 m and the goal sphere 0.0 m from their commands; after the commands moved, a
render with no callback put both on the new point (0.0 m).

| image | subgoal family on | goal family on | real sphere at the same point |
|---|---|---|---|
| policy depth, 80×45 | identical to off, both cycles | identical to off, both cycles | 79 cells change by more than 0.1 m |
| policy grid, 3 600 cells | identical | identical | 79 cells |
| perception depth, 640×360 | identical | identical | 5 057 pixels |
| policy RGB | 83–90 cyan pixels (0 when off) | 126–131 yellow, 4–6 azure (0 when off) | 102 magenta |
| perception RGB | about 5 900 cyan (0 when off) | about 7 950 yellow, about 960 azure (0 when off) | 4 723 magenta |

Re-rendering one state repeats every depth image exactly, and removing the sphere restores the
baseline exactly, so "identical" is a live comparison. The frames show it directly
(`check2/run/frames/`): the cyan subgoal sphere, its cone and the path dots, and the yellow goal
sphere with its azure cone, in the RGB of both cameras, and nothing in either depth.

**What it means.**
- The pre-flip depth policies trained with positioned markers in every environment and, on the
  camera measured here, none in their depth input. `debug-marker-leak-2026-09-21` §3 listed
  "reached its D555" as inferred for them. It is now measured for the 80×45 camera of v1, v2,
  v2.1, v2.1a and cprime: RGB yes, depth no. v0 and the warm-start legs used an 80×60 camera (a
  4 819-dim observation) that was not rendered here.
- **So a marker was never in a pre-flip depth policy's input, and cannot have caused any v0–cprime
  result through it.** The depth policies' observation contract is 19 scalars and the depth
  image — 3 619 dims at 80×45, 4 819 at v0's 80×60 — with no RGB field in any variant
  (`source/strafer_shared/strafer_shared/policy_interface.py`; the exported models record 4 819
  for v0 and 3 619 for v1 and v2). Check 2 puts the pre-flip markers in RGB only. For v1 onward
  the depth half is measured; for v0 and the warm-start legs it rests on the same marker code.
- RGB has other readers: the bridge's perception stream feeds the Jetson's perception stack,
  including the grounding the language missions use. Whether a marker reached those is a
  separate question and stays open.
- This measures the Isaac Sim 6.0.0.0 and `IsaacLab-retired` install on this host, which predates
  every pre-flip run; no record pins the Isaac Lab checkout per training run.
- On the pre-flip stack the bridge's goal marker reached the perception stream's RGB and never
  its depth, which fits the bridge capture of 2026-08-22 showing no marker in its depth frames.
- The markers reached depth only on the post-flip stack, where the leak record measured them.
  Which change between the two stacks did it is not isolated here; the marker prototypes carry
  `invisibleToSecondaryRays` on both.

## 2. Check 4 — v3 with the markers positioned, against none

**Setup.** Tree `2575c65`, the #223 merge, on the canonical pair. Its code is that of `fa4cb93`,
the last `main` before #224 removed the marker geometry; the #225 merge between them changes one
record. The run carries one scratch commit (`check4/check4_eval_flag.patch`, not merged). It adds
`--no_command_debug_vis`, which sets `goal_command.debug_vis=False` before the env is built, and a
probe that reads the marker instancers from the stage at steps 1, 60 and 300 without touching
the rollout. The protocol is G7's: `eval_cadence_emulation.py` on
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, `--profile clean --num_envs 16
--episodes 100 --seed 42`, one Kit boot per sample, four per arm, alternating ON and OFF.

| arm | flags | what the probe shows, every run |
|---|---|---|
| ON | `HEADLESS=1 --viz kit` | 16 subgoal spheres and 16 cones, each at its environment's pre-step subgoal with a difference of exactly 0.0 m; none at the world origin; `KitVisualizer` registered |
| OFF | the same, plus `--no_command_debug_vis` | no command marker prims |

On the current pair G7's `--headless` registers no visualizer, so its samples there never
positioned the markers; on the previous pair they were positioned all the same, into RGB only
(§1). Both arms here carry the Kit visualizer, so they differ only in whether the markers exist.

**v3 per run** (`check4/summary.txt`):

| run | ON completion | ON offset median | ON left | OFF completion | OFF offset median | OFF left |
|---|---|---|---|---|---|---|
| 1 | 0.840 | 4.61° | 0.610 | 0.920 | 1.33° | 0.530 |
| 2 | 0.920 | 4.72° | 0.611 | 0.880 | 1.04° | 0.527 |
| 3 | 0.873 | 5.78° | 0.616 | 0.910 | 0.97° | 0.521 |
| 4 | 0.920 | 4.72° | 0.611 | 0.910 | 1.84° | 0.543 |

**v3 arm means** (sample standard deviation over four runs; the ON arm's rests on three distinct
rollouts, below). Standard errors are unpooled (Welch): √(sd_ON²/4 + sd_OFF²/4), as
`check4/check4_summary.py` computes them.

| metric | ON | OFF | ON − OFF (standard error) |
|---|---|---|---|
| completion (= path_complete) | 0.888 ± 0.039 | 0.905 ± 0.017 | −0.017 (0.021) |
| direction-offset median | 4.96° ± 0.55 | 1.29° ± 0.40 | +3.66° (0.34) |
| direction offset, median of the absolute | 14.96° ± 1.05 | 14.96° ± 0.38 | −0.00° (0.56) |
| direction offset, p10 / p90 | −31.0° / +53.1° | −40.6° / +68.1° | +9.5° / −15.0° |
| fraction left | 0.612 ± 0.003 | 0.530 ± 0.009 | +0.082 (0.005) |
| sustained collision | 0.099 ± 0.038 | 0.093 ± 0.019 | +0.007 (0.021) |
| off-path divergence | 0.007 ± 0.009 | 0.003 ± 0.005 | +0.005 (0.005) |
| near-arrival | 0.522 ± 0.055 | 0.500 ± 0.035 | +0.022 (0.032) |
| progress, mean | 0.907 ± 0.019 | 0.896 ± 0.019 | +0.011 (0.013) |

No run flipped the robot; time-outs were 0 or 0.01.

- **ON runs 2 and 4 are the same rollout.** Their episode lists are identical, so the ON arm
  holds three distinct rollouts. A same-seed launch can replicate another exactly; G7's eight
  did not. Dropped, the difference is −0.027 (0.025) in completion and +3.74° (0.42) in offset,
  and the reading does not change.
- **The reading.** What was written down before the runs (`check4/PREREGISTRATION.md`) called v3
  indifferent if |Δ completion| ≤ 0.04 and |Δ direction-offset median| ≤ 0.5°. Completion passes.
  The offset does not: the difference is 10.8 standard errors from zero and 9.3 past the
  threshold, or 8.9 and 7.7 with the duplicate dropped. With markers in view v3's median command
  moves left of its subgoal and its offsets spread less — the p10-to-p90 range is 82–89° in every
  ON run against 102–113° in every OFF run — while its median absolute offset stays at 14.96°. No
  outcome metric moves beyond its standard error in either direction: completion, collisions,
  off-path divergence, near-arrival and progress — though four runs at standard error 0.021
  cannot exclude a completion effect of about 0.04. So this is a change of steering, and neither
  criterion improves with markers. Why v3 steers this way is not tested here; its own training
  kept the markers at the world origin, so a marker at its subgoal is a novel object to it.
- **The band could not certify indifference.** The maintainer wrote the 0.04 / 0.5° band into
  the reading before the runs. Against the standard error of a four-run difference built from
  G7's own spread (0.026 in completion, 0.40° in offset) it is 1.5 and 1.3 standard errors wide,
  so a true null falls outside the offset band about a fifth of the time (21 %) and outside at
  least one of the two about 31 % of the time. The +3.66° observed here, 9.3 standard errors past
  the band, is a real effect regardless.

**The v2 control**, the same two arms on the same tree, the same day:

| metric | ON | OFF | ON − OFF (standard error) |
|---|---|---|---|
| completion | 0.010 ± 0.012 | 0.093 ± 0.013 | −0.083 (0.009) |
| direction-offset median | 31.3° ± 1.8 | 8.1° ± 2.3 | +23.1° (1.4) |
| off-path divergence | 0.793 ± 0.028 | 0.465 ± 0.038 | +0.328 (0.023) |
| progress, mean | 0.232 ± 0.011 | 0.500 ± 0.013 | −0.268 (0.008) |

- OFF runs 1 and 2 are the same rollout.
- v2 collapses on this tree with or without markers: 0.093 completion off, against 0.86 in G7.
  The env moved under it after G7 — the near-field convention (#219), the delay buffer (#220),
  the noise band (#221) and the deploy-resolution depth path (#222) — so v2 is compared ON
  against OFF here, never against G7.
- Markers make it far worse. By check 2, v2 never had a marker in its depth, so here they are
  new near-field objects to it, as they are to v3.

## What is not claimed

- Why the post-flip stack draws the markers into depth and the pre-flip one does not.
- Why v3's steering changes with markers in view.
- The Isaac Lab checkout each pre-flip policy trained on, which no record pins per run.
- Anything about v2 on the tree it trained on: the control ran on `2575c65`.
- That v3's leftward bias matters anywhere a command marker cannot be rendered, which since #224
  is everywhere.
- Anything about the real D555.

## Deviations from the protocol as first specified

- G7's `--headless` is replaced by `HEADLESS=1` and `--viz kit` in both check-4 arms, because
  positioning the markers needs a visualizer on this stack.
- One v2 ON launch gave up after three boot stalls and wrote nothing; it was relaunched
  (`check4/logs/launch1_stalled.*`). Every other launch finished, five of them after one or two
  boot stalls; check 2 booted first time.
- Check 2 measures in place rather than driving a policy, adds the perception camera and the
  bridge's goal term to v2's task cfg, and builds the room without a ceiling so the RGB is lit.
  In its on/off cycles the marker callback is called directly; proofs A and B show the stack
  positioning the markers on its own.
- v2's checkpoint is the preserved copy in `isaac-lab-upgrade-baseline-2026-08-14` §6, the one G7
  used; `goal-a-attribution-2026-08-22` preserved only the ONNX export.
- `check4/PREREGISTRATION.md` describes `2575c65` as the last `main` with marker geometry. It is
  the #223 merge; `fa4cb93` is the last such `main`, with the same code.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `debug-marker-checks-2026-09-23/record-files/` |
| deposit commit | `91606f77ef29b8a463db90b06a7da2713f672d4b` |

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/debug-marker-checks-2026-09-23/record-files/. \
      docs/measurements/debug-marker-checks-2026-09-23/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/debug-marker-checks-2026-09-23/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
9d82d694e69fce5c1de8c9f32b0b6194c2e915509fa517bd33124e37bfbc8926  check2/PREREGISTRATION.md
1b49900b5b1737a129b653e2112f65f2b08709a344bdc8e2050aafa655481b76  check2/d555_marker_visibility_preflip.py
2b18311b89f65cae0848cf9b73efcb342b591715e22ea9cc2d84a01472341d5a  check2/d555_marker_visibility_preflip_cmd.sh
08b4663b1c5c1a79e5c78da4975834aa87f92f6688b65550f9b084cb7d2f0236  check2/run/d555_marker_visibility_preflip.json
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/a0_baseline_perception_depth.png
476ef0fbd427a9b348ee6466be53ba6b5da52e0c502da14b5df5810cd13549a0  check2/run/frames/a0_baseline_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/a0_baseline_policy_depth.png
6c1caa732349e0755dcdbb2a7c9b0fe12b4655ffd8a5e4fa92094a8c9dc6a540  check2/run/frames/a0_baseline_policy_rgb.png
760527157ab7f18c9f831394c7b18d2eeeb4d846716925b655933231aaeab8aa  check2/run/frames/a0_control_perception_depth.png
9a95a0aae0890a775f52c98c051b9a4bb2350d64af00b557ba258fcddc8f2edf  check2/run/frames/a0_control_perception_depth_absdiff_vs_baseline.png
9360438737f16f0d8574edd979613dec2a949c9aa1fd2a3d6db5c9e06a0301a9  check2/run/frames/a0_control_perception_rgb.png
55f42c9df6539f9cc340f32dcd6544b606d3e7bdb7a0ae1afac0e9a8d3d95382  check2/run/frames/a0_control_policy_depth.png
d72e42c5735ed33ae02dcf5fd345e8dc52b6958fc02d519245318fb84bb0aea8  check2/run/frames/a0_control_policy_depth_absdiff_vs_baseline.png
56f76ae111faef603c83510eca607485ddcc76ed17a383e6df93d07236cf49e5  check2/run/frames/a0_control_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/a0_natural_both_on_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/a0_natural_both_on_perception_depth_absdiff_vs_baseline.png
2dbdd0dcc3c1b115129b9b68a4247be25127b148ffc188399dcc5236c80f55a5  check2/run/frames/a0_natural_both_on_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/a0_natural_both_on_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/a0_natural_both_on_policy_depth_absdiff_vs_baseline.png
467c22d0ff8025b8a4bb8aa6b7c4336fac6296f77f93d80fa1e37a11e8e99208  check2/run/frames/a0_natural_both_on_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/goal_off1_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_off1_perception_depth_absdiff_vs_baseline.png
d013b1d58c8d23ef783f8f89ff197130f86942456715d536b97ce8998665f2a8  check2/run/frames/goal_off1_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/goal_off1_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_off1_policy_depth_absdiff_vs_baseline.png
11b1734bc413f16e29fe892830f1aa3998bfb5b27222b1df472f6e22a832d0d5  check2/run/frames/goal_off1_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/goal_off2_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_off2_perception_depth_absdiff_vs_baseline.png
243744d15efbaf4db1e6ced8b4f8b59b80337efe10397d2b0c429ee2f4541a09  check2/run/frames/goal_off2_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/goal_off2_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_off2_policy_depth_absdiff_vs_baseline.png
89a103255a7a12ff5ff87abf11e1e2a869eafbf872dfbb474f4b125e1ae4f5b6  check2/run/frames/goal_off2_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/goal_on1_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_on1_perception_depth_absdiff_vs_baseline.png
16891f62f6c7eb2907563d672ee49f7e4992837070f7a221046b83d80a7e9d57  check2/run/frames/goal_on1_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/goal_on1_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_on1_policy_depth_absdiff_vs_baseline.png
7b54186a5c2b1bbdfb863653fe3076cfe951c0bd4a602deadd06ea4c68337187  check2/run/frames/goal_on1_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/goal_on2_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_on2_perception_depth_absdiff_vs_baseline.png
de0920ce154c46ed9b2a82d52e443d8ca4c32d216352b05b8babfcafcb1f5a09  check2/run/frames/goal_on2_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/goal_on2_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/goal_on2_policy_depth_absdiff_vs_baseline.png
c3a3a9f408459897200ca4ff333fbd15a6e32a9c167bcdba3e152877cf071a11  check2/run/frames/goal_on2_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/subgoal_off1_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_off1_perception_depth_absdiff_vs_baseline.png
354944e011f02c72248fcf0a32526fcfce1c9629e02c9bf1e04637ccd82e7984  check2/run/frames/subgoal_off1_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/subgoal_off1_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_off1_policy_depth_absdiff_vs_baseline.png
06edd1037216e58671b8cbb4de626e505c2b295a0757d6aca6d9f4bcf85c50a9  check2/run/frames/subgoal_off1_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/subgoal_off2_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_off2_perception_depth_absdiff_vs_baseline.png
02835e6fe3f5f2665e2edcc9277a5e4182b48059e27b82ec0dc79319cbd1af3d  check2/run/frames/subgoal_off2_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/subgoal_off2_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_off2_policy_depth_absdiff_vs_baseline.png
e237073d3f183db635e33539f51871c4883c630d198c1b4583fdeda55d4debcd  check2/run/frames/subgoal_off2_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/subgoal_on1_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_on1_perception_depth_absdiff_vs_baseline.png
40c3fc9e786f1957c1fdfdc8454bccf481840e4061b056e08c2b3b4ac7ce1b7a  check2/run/frames/subgoal_on1_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/subgoal_on1_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_on1_policy_depth_absdiff_vs_baseline.png
d2c84847aaf0d3853ffb84d9782b3558f01bbb04679195c92680c4daa77c01ce  check2/run/frames/subgoal_on1_policy_rgb.png
f8cf387c5d9bd3f14612d8214989fd570df8ee666bafb5cc86d9eccb9990f52b  check2/run/frames/subgoal_on2_perception_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_on2_perception_depth_absdiff_vs_baseline.png
6008a3892c2b7a6d34489d06e7c711aa81708c7c9a686808f1870e4001edf79b  check2/run/frames/subgoal_on2_perception_rgb.png
758a2f44dd5e96fea267d38f08549bcb325d9606aba37fb2c1348ffe34a9c944  check2/run/frames/subgoal_on2_policy_depth.png
c52650a3c12eed8846c0c2f31934b4dd7a58b506ca26d581fd4e3a70fc8a603b  check2/run/frames/subgoal_on2_policy_depth_absdiff_vs_baseline.png
f39e3db79ced945f9ad5faec9fb649cc2958a91719d564cf720aa288a7d90c43  check2/run/frames/subgoal_on2_policy_rgb.png
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check2/run/gpu_after.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check2/run/gpu_before.txt
554c5ac9666f69424c0b1a83e29a219f30ff8071ce3f91004cb075040aa3df4a  check2/run/preboot_resolution.txt
407923c859d3467132580bbc8f6a664fe42eeefca4c1b047e80286bef9a7bd5c  check2/run/probe.log
dbc856840bfdf641965601b0231ba042b6ddd25f8f42d55e11bb3abdafb823e5  check2/run/watchdog_check2.log
dbc856840bfdf641965601b0231ba042b6ddd25f8f42d55e11bb3abdafb823e5  check2/run/watchdog_check2.log.attempt1
f6f9e468910981f9b46d818747866e9a4fd58db9da9859d1a3500961de669f72  check4/PREREGISTRATION.md
f696361a6fa13ea3f3d78565831f8da0691927d10a3a249f4ce1d869bd4c17ba  check4/check4_eval_flag.patch
ff351595d1efa6b45d8e8b9f3965e5b5fb11258459acc50924766e9976c22b02  check4/check4_run.sh
3e99279b7d56d0c23512079d31579879895984dd741eff5a005be9d1e9b723ba  check4/check4_series.sh
567f9fa9e0a376671caefb3daf30651c8915d3e3f98ad05c829a4bb643b39f85  check4/check4_summary.py
e1e4a5a22afa87ce146e3dbd6aba84a84162aeff920bae1995c0d4eab4cfacb9  check4/checkpoints_sha256.txt
c68a5e104f4f355abcd8c5d4e8e091165dd27a040bb18448355c9ebced61a06d  check4/eval/v2_off_run1/cadence_20260923_230807.jsonl
c68a5e104f4f355abcd8c5d4e8e091165dd27a040bb18448355c9ebced61a06d  check4/eval/v2_off_run2/cadence_20260923_231901.jsonl
d9d646fa3201a9c7e2d11c8352667022e911000ca1bc4971676454574a9ef35d  check4/eval/v2_off_run3/cadence_20260923_233026.jsonl
7e4b35782c75014a001f2c3157e14ab962aac60239c15b1c422d86aefe78fae8  check4/eval/v2_off_run4/cadence_20260923_234502.jsonl
7330e250cfff755247cc1795da265c3c66b09e862a068e3d201b24b971e20597  check4/eval/v2_on_run1/cadence_20260923_230231.jsonl
48c0d8828ea0010fe2e8bc2da958b093e981153012f237c0c8038f4882d01c4f  check4/eval/v2_on_run2/cadence_20260923_231224.jsonl
6a96736b2322a4d58ddb13fa0f71af7faa2b5ad7982537a2c7c382ecfc06f281  check4/eval/v2_on_run3/cadence_20260923_232314.jsonl
9fbb7234086477daa6a0c5baeb4133271f75cffe2da222f67c58872004a4f3f0  check4/eval/v2_on_run4/cadence_20260923_233914.jsonl
44b3a7edaa9db6e06f09a0573a711fb8431b3581ef5684bcae4a772f72424520  check4/eval/v3_off_run1/cadence_20260923_223839.jsonl
6d7f9ca717787cd19a14f1bdc5d599cdd0ea22b53d57eaea54a8e9cd12884165  check4/eval/v3_off_run2/cadence_20260923_224428.jsonl
91fa5fefda17165436658f44b2f5d3a54834cbdd8926d4e938309e42a776772e  check4/eval/v3_off_run3/cadence_20260923_225223.jsonl
736829a240ab729c2bc8d9d352a6839cd7a4050914ea177a5185d8f941105d9a  check4/eval/v3_off_run4/cadence_20260923_225818.jsonl
3589c20709a6075878aa1e871dc71098ffcc56dba62af87a6cd6d550fbcf840e  check4/eval/v3_on_run1/cadence_20260923_223518.jsonl
dc6bfa4bc994730021df5336b11079742b2faa2fe52812945a18b2a7c979a1d3  check4/eval/v3_on_run2/cadence_20260923_224151.jsonl
85e75ce8b19270db5e553816d4422febbbb6526ba37308caf2d2c09b27df9424  check4/eval/v3_on_run3/cadence_20260923_224935.jsonl
dc6bfa4bc994730021df5336b11079742b2faa2fe52812945a18b2a7c979a1d3  check4/eval/v3_on_run4/cadence_20260923_225532.jsonl
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/launch1_stalled.v2_on_run4.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/launch1_stalled.v2_on_run4.gpu_before.txt
73111f6430740e05591a8b800d59177fc79318868cd4cba05dc10c81ee91888a  check4/logs/launch1_stalled.v2_on_run4.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/launch1_stalled.watchdog_v2_on_run4.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/launch1_stalled.watchdog_v2_on_run4.log.attempt1
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/launch1_stalled.watchdog_v2_on_run4.log.attempt2
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/launch1_stalled.watchdog_v2_on_run4.log.attempt3
506dfaadba17921cd1abba5361357eba79a41d5762af235275f8008b8afeb573  check4/logs/sampling.log
fb77f30e1f40408752d4ef3e30331eb046cd319b364830ceef3e44a4ade079ec  check4/logs/series.out
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_off_run1.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_off_run1.gpu_before.txt
d43205eef009293090befbe23f198fd0ce7d8f50ad2d7de25748df553a94b51a  check4/logs/v2_off_run1.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_off_run2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_off_run2.gpu_before.txt
04ad6932e6e57fede09c220d36b4ec5b95654905fa18f7eebe93de2a85e416a9  check4/logs/v2_off_run2.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_off_run3.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_off_run3.gpu_before.txt
78ca23903713d9173abe52ce0c1c71f4ce55f71661970e006316786e2b72d5a7  check4/logs/v2_off_run3.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_off_run4.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_off_run4.gpu_before.txt
f60a912f36436944662048ed74925d9c10b52b4577fd0cbf729c174a8a6a194c  check4/logs/v2_off_run4.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_on_run1.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_on_run1.gpu_before.txt
fd14f5481ac1a4da3b70ead1b6b1849f5c390f2135ae9ca152a8684a3079ba58  check4/logs/v2_on_run1.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_on_run2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_on_run2.gpu_before.txt
19af507540df7d19c67277b64363cbb4136c6b505e13c6e5342596b878c67636  check4/logs/v2_on_run2.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_on_run3.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_on_run3.gpu_before.txt
2dd1c1235f98cbaab29f805801f4e42bb883c1058ba3cbabf90cbda0f409cc74  check4/logs/v2_on_run3.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v2_on_run4.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v2_on_run4.gpu_before.txt
a427a76ce7d9d551066f9f2f5cc3fa7591335ff8906260edaff6cea922a70e36  check4/logs/v2_on_run4.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_off_run1.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_off_run1.gpu_before.txt
c9a7730b13c748376c19a2522dcd470568f9a820b58bd9969f8add4f56134e10  check4/logs/v3_off_run1.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_off_run2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_off_run2.gpu_before.txt
ebbbb735ee4d56f93421c276cc4a419bbcdf0a8f0bf798b8aff3c1e48541dc94  check4/logs/v3_off_run2.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_off_run3.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_off_run3.gpu_before.txt
d7c9616689d0c1a7915e4922657714357b0ea82ea7619f38b4487bb369f44baa  check4/logs/v3_off_run3.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_off_run4.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_off_run4.gpu_before.txt
c86515ced665cf63a94e61cf14c2954da8ddac311a1cf6bb5847c995e48ecdcd  check4/logs/v3_off_run4.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_on_run1.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_on_run1.gpu_before.txt
b3b2020cf338a899d203216887756593ee851fa0fe9a045a19ca727915a5e880  check4/logs/v3_on_run1.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_on_run2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_on_run2.gpu_before.txt
92a5306635863c1351d4d3cab268b2d3416d4e351f186eb2a19b60f9b5e4efcd  check4/logs/v3_on_run2.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_on_run3.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_on_run3.gpu_before.txt
980c0600613dfb46ad67b6110adef30465f340a0bb5913303302ea50ac472405  check4/logs/v3_on_run3.log
4287bbc0c5e1a78af0153b801b91e5ea33974360947060b2b0d1c8026cfde80b  check4/logs/v3_on_run4.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  check4/logs/v3_on_run4.gpu_before.txt
0421e1718730c319955e0f3fa39b728671f385b692e94d4f3b7792ba6207d796  check4/logs/v3_on_run4.log
9f885743f6e7a2f302bb36e0745dfd68c576d53a0dfb63c799b7e1b19e8a47c9  check4/logs/watchdog_v2_off_run1.log
9f885743f6e7a2f302bb36e0745dfd68c576d53a0dfb63c799b7e1b19e8a47c9  check4/logs/watchdog_v2_off_run1.log.attempt1
90f197e9ce3ff8f0216d0311dc075635cb67305402956093e4aabf5b471b57ec  check4/logs/watchdog_v2_off_run2.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v2_off_run2.log.attempt1
90f197e9ce3ff8f0216d0311dc075635cb67305402956093e4aabf5b471b57ec  check4/logs/watchdog_v2_off_run2.log.attempt2
07036cefafb1cf28c344644454c13b1a9556b18201a6ffbaef09e4bf8eb17cf8  check4/logs/watchdog_v2_off_run3.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v2_off_run3.log.attempt1
07036cefafb1cf28c344644454c13b1a9556b18201a6ffbaef09e4bf8eb17cf8  check4/logs/watchdog_v2_off_run3.log.attempt2
164cc70c27b8835dcc11536c04e0e5a38add45b85cecdd156218ed777b7664d6  check4/logs/watchdog_v2_off_run4.log
164cc70c27b8835dcc11536c04e0e5a38add45b85cecdd156218ed777b7664d6  check4/logs/watchdog_v2_off_run4.log.attempt1
9648e397af6789e6dee11335328efd6980e8e53b8349e253a901b97f3549e3a5  check4/logs/watchdog_v2_on_run1.log
9648e397af6789e6dee11335328efd6980e8e53b8349e253a901b97f3549e3a5  check4/logs/watchdog_v2_on_run1.log.attempt1
6f3a276ef3166ecea358605b11d155dc4168edb8da3f6d8234e5f8872f6c570e  check4/logs/watchdog_v2_on_run2.log
6f3a276ef3166ecea358605b11d155dc4168edb8da3f6d8234e5f8872f6c570e  check4/logs/watchdog_v2_on_run2.log.attempt1
141c9536e5c76d3046a6f4437b6cf357ca7e8b417f691a5d5f33c53c87ee362c  check4/logs/watchdog_v2_on_run3.log
141c9536e5c76d3046a6f4437b6cf357ca7e8b417f691a5d5f33c53c87ee362c  check4/logs/watchdog_v2_on_run3.log.attempt1
3173e59ef06e1f6279d220a13ca5cc1c67506687045dd6d109a99c7fb19ac9d1  check4/logs/watchdog_v2_on_run4.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v2_on_run4.log.attempt1
3173e59ef06e1f6279d220a13ca5cc1c67506687045dd6d109a99c7fb19ac9d1  check4/logs/watchdog_v2_on_run4.log.attempt2
1b04536b5fb00171114e5aea70b02558aa2ad8f317d6dd3e648f23355c8c0ab7  check4/logs/watchdog_v3_off_run1.log
1b04536b5fb00171114e5aea70b02558aa2ad8f317d6dd3e648f23355c8c0ab7  check4/logs/watchdog_v3_off_run1.log.attempt1
1742fb45c705696cde8b9849434814d7c2384657b77bc4fe4c04e34daa930d59  check4/logs/watchdog_v3_off_run2.log
1742fb45c705696cde8b9849434814d7c2384657b77bc4fe4c04e34daa930d59  check4/logs/watchdog_v3_off_run2.log.attempt1
d479d68a9fe5ae121d2737c845c8bffb6d6c2bd486d042934eb88c742c08ace8  check4/logs/watchdog_v3_off_run3.log
d479d68a9fe5ae121d2737c845c8bffb6d6c2bd486d042934eb88c742c08ace8  check4/logs/watchdog_v3_off_run3.log.attempt1
13ae10d55aae48f7d38f7d6bcc380fcb5b695b69c79f2bde0952b535432ce428  check4/logs/watchdog_v3_off_run4.log
13ae10d55aae48f7d38f7d6bcc380fcb5b695b69c79f2bde0952b535432ce428  check4/logs/watchdog_v3_off_run4.log.attempt1
e57c2ee609cd5b829bdebcb51fe6f3e10d680c8f01c03476223beb2e43274e55  check4/logs/watchdog_v3_on_run1.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v3_on_run1.log.attempt1
e57c2ee609cd5b829bdebcb51fe6f3e10d680c8f01c03476223beb2e43274e55  check4/logs/watchdog_v3_on_run1.log.attempt2
80101f95c4536b795655f703569861c71d2da0fca22df888e9f00fe8d1b74ac6  check4/logs/watchdog_v3_on_run2.log
80101f95c4536b795655f703569861c71d2da0fca22df888e9f00fe8d1b74ac6  check4/logs/watchdog_v3_on_run2.log.attempt1
215390105ab887a566c033057b385b219d098ff7b3b159c07d02f2593efd1a2c  check4/logs/watchdog_v3_on_run3.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v3_on_run3.log.attempt1
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  check4/logs/watchdog_v3_on_run3.log.attempt2
215390105ab887a566c033057b385b219d098ff7b3b159c07d02f2593efd1a2c  check4/logs/watchdog_v3_on_run3.log.attempt3
52c393d102c6cce2c173234dca5e00f25ca91442d1dda784b63fa2526a66a1d6  check4/logs/watchdog_v3_on_run4.log
52c393d102c6cce2c173234dca5e00f25ca91442d1dda784b63fa2526a66a1d6  check4/logs/watchdog_v3_on_run4.log.attempt1
b1ffeec83aec61788ce0352568b793c9adc4f63f24ae6492c25fcb6b5948c8dd  check4/summary.txt
```
