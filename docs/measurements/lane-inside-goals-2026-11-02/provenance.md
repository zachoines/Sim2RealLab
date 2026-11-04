# Provenance — the policy at inside-room goals on the lane, under A‴, 2026-11-02

captured_date: 2026-11-03 (registered 2026-11-02). IB1 ran 2026-11-02 14:14–14:34 UTC; IB1r, IB2 and IB3 ran
2026-11-03 22:08–23:31 UTC. Paths are relative to the deposit directory unless they name the main repository.

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10, aarch64) | 7.0.0-1019-nvidia | Driver 580.173.02. Runs the bridge, the planner and VLM services and the session's drivers (`dgx/`) as systemd user units, with logs on persistent disk; the 1 Hz telemetry (`dgx/fh_telemetry.py`) and, from IB1r on, the 1 Hz memory log (`campaign/meminfo_log.py`). |
| robot | strafer-nx | not read in this deposit | Runs the deploy stack in five containers (`session/fh_up.sh`), the reach harness, stage 0 (`session/stage0_fh.sh`), the frame check (`tools/frame_check4.py`) and the recording. Hardware as read on 2026-10-10 for `goal-a-cli-video-2026-10-09`: a Jetson Orin NX 16 GB, L4T R36.4.3. |

- **The link.** The direct link carried DDS and the planner and VLM calls; the sim host drove each boot over it by ssh.
- **The robot host was held** for this campaign from 2026-11-02 14:08:45 UTC, after its other work had cleared it, to
  its release at 2026-11-03 23:33 UTC; the hold closed at 23:34 UTC (`campaign/host_hold.log`).
- **Under the hold, before IB1:**
  - a read-only feasibility read and a QoS check, in transient containers (`campaign/viz_feasibility.txt`,
    `campaign/viz_qos_check.txt`);
  - the a3 session's files on the robot host moved aside;
  - this deposit's `session/`, `tools/` and `goals/` copied in, hash-equal (`campaign/robot_files_sha256.txt`).

## Trees and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `0a975ef` (`git=0a975ef5` in each `bridge.log`). Its `strafer_lab`, `strafer_autonomy` and `strafer_vlm` equal `main`'s at `f16d18d6`. | conda `env_isaaclab3` (Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1); the planner and VLM services from the same checkout (`campaign/services/`), started 2026-11-02 14:12:57 UTC and, after the power cycle, 2026-11-03 22:06:46 UTC (`campaign/units_journal.txt`); stopped by their process groups after IB3 |
| robot | images labelled `f16d18d64c2f`, built 2026-11-02; the checkout at `6715cf3` with the lane's two env-file edits (stage 0's "robot checkout" lines), and three compose files from outside the tree (`campaign/robot_host_checks.txt`) | `strafer-gpu` `sha256:659b7649552a…` (`inference`); `strafer-cpu` `sha256:a40d973431e8…` (the other four) (`boots/<RUN>/robot/stage0/images.txt`, identical in all four segments) |

**rtabmap, read from inside `strafer_slam` in every segment** (stage 0, `images.txt`; identical in all four, and equal
to the a3 session's):

| package | build |
|---|---|
| `ros-humble-rtabmap` | `0.23.13-1jammy.20261002.091850` |
| `-slam` | `0.23.13-1jammy.20261008.162341` |
| `-ros` | `0.23.13-1jammy.20261008.165507` |
| `-launch` | `0.23.13-1jammy.20261008.165222` |

**The parameters.** `rtabmap_params.yaml` inside the image has sha256 `82025bc7…`, with no mount over it, at every
bring-up (`boots/<RUN>/robot/logs/rtabmap_params_check.txt`). **The policy** is
`strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd5…`, sidecar `d22e3504…` (`campaign/robot_host_checks.txt`). Every
bring-up and recreate logged `vel_cap=(0.7841 m/s, 2.0933 rad/s) policy_loaded=True`.

## Segments

| segment | token | bring-up (UTC) | boot done | database (sha256, size) |
|---|---|---|---|---|
| IB1 | `ig1IB1` | 2026-11-02 14:14:36 | none: the sim host froze at about 14:34:43 | `3fc4ff1e…`, 191,004,672 B |
| IB1r | `ig1IB1r` | 2026-11-03 22:08:36 | 22:24:44 | `0613058c…`, 154,972,160 B |
| IB2 | `ig1IB2` | 2026-11-03 22:25:42 | 22:58:56 | `f1719eeb…`, 321,327,104 B |
| IB3 | `ig1IB3` | 2026-11-03 22:59:48 | 23:31:34 | `d1b655fe…`, 311,984,128 B |

- **The databases and the recording's bags** are kept on the sim host, with their full sha256 in
  `campaign/databases_sha256.txt`.
- **IB1's teardown** ran on 2026-11-03 at 22:04 UTC (`boots/IB1/dgx/down.txt`).

## The sim host's freeze

- **Telemetry** (`campaign/telemetry.csv`): its last line is 2026-11-02 14:34:43 UTC.
- **The journal's last line** is 14:31:55 UTC (`journalctl --list-boots`, `campaign/journal_reads.txt`).
- **The kernel** holds one GPU-driver line besides the module's load, read with `journalctl -b -1 -k`
  (`campaign/journal_reads.txt`): `NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY]
  (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359`, at 14:14:27 UTC.
- **Power-cycled** 2026-11-03 at 16:40 UTC.
- **Before each later bridge launch** the page cache was dropped (`campaign/evict.log`, in GiB though labelled "GB":
  IB1r 17.17 → 1.87, IB2 4.91 → 2.09, IB3 4.56 → 2.09). The boot that ran them has no out-of-memory or Xid kernel line,
  read after IB3 (`campaign/journal_reads.txt`). `getcap` on py-spy was empty and `ptrace_scope` 1
  (`campaign/getcap_preboot.txt`), as registered.

## Inputs

| input | sha256 | notes |
|---|---|---|
| `PREREGISTRATION.md` | committed at `a7df5b0`; additions `bd9c3ae`, `01c6d30` | before IB1 |
| `OPERATIONAL_NOTE_2026-11-03.md` | committed at `cffc8ad` | before IB1r's first reach |
| `goals/goals.json`, `goals/inside_goals.json` | `89d7ee19…`, `ddf4c2c5…` | the draw and the chain's copy |
| `tools/to_start.py` | `9cffcbf4…` | v2 |
| `frame_health.py` | `d8792cd2…` | unchanged from the a3 deposit |
| `ttg_summary.py` | `a2b7c519…` | the baseline deposit's, unchanged |
| `inside_reads.py`, `odom_terminal.py`, `slam_reads.py`, `arm_summary.py`, `reproduce.sh` | `1b027e6d…`, `06d3530b…`, `fc405ea1…`, `8c732d2e…`, `782a620f…` | as registered |
