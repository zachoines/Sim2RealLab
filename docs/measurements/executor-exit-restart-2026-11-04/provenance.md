# Provenance: the executor's exit cancels its policy goal, 2026-11-04

captured_date: 2026-11-04 UTC (2026-11-03 CST)

## Hosts

| role | host | system | kernel |
|---|---|---|---|
| every container, probe and test | strafer-nx | Jetson Orin NX 16 GB, L4T R36.4.3 | 5.15.148-tegra |
| the Isaac bridge, planner and VLM for §1 | the sim host | DGX Spark | |

- **No hardware in any run.** `base` and `perception` did not run, no RoboClaw was on USB, and no
  container bound `/dev`.
- **The robot host was held** by session strafer-nx from 00:41:13 until it released the host at 01:03:19
  (`host_hold.log`). The sim host's side, from the request to the stop, is `dgx/host_hold.log`.
- **The lane check** ran on host networking on `ROS_DOMAIN_ID` 42, the lane's domain, with the
  rig's direct-link Cyclone config.
- **The fence probes** ran on host networking on `ROS_DOMAIN_ID` 77, with `ROS_LOCALHOST_ONLY=1`
  except the three as-deployed runs. The signal probes ran with `--network none`. The probes after
  01:03 ran after the host was released, with no stack on either host.

## Images

| use | image |
|---|---|
| §1, the lane check | built at `efd4e7e52a1f` by `make images` (`build/make_images.log`): `strafer-cpu` `sha256:09206258885a…`, `strafer-gpu` `sha256:0fdfa153c046…` |
| §2–§4, the probes and the test runner | `strafer-cpu:humble` `sha256:a40d973431e8…`, label `f16d18d64c2f`, with the tree's sources mounted over the image's (the fence probes put the tree's `strafer_inference` first on `PYTHONPATH`; the signal probe's fixed run did the same for `strafer_autonomy`) |

The registered images, `strafer-cpu` `a40d973431e8` and `strafer-gpu` `659b7649552a`, were kept
under extra tags through the build. `:humble` was restored to them at 01:03:08, and the fix images
stay tagged `:efd4e7e` (`build/restored_tags.txt`). The rosdep layer was rebuilt, and
Cyclone is ros-humble-cyclonedds 0.10.5 in both images.

## Trees

| run | tree |
|---|---|
| `tests/test_on_main_*.txt` | `main` at `f16d18d6`, with this branch's two test files added and nothing else |
| `probes/signal/`, `main` row | the image's installed executor, built from `main` at `f16d18d6` |
| `probes/signal/` fix rows, `probes/fence/` 00:36–00:37 | the working tree before the work commit, whose code files are the work commit's (the README edits came after) |
| `probes/fence/` 00:46–00:47 and 01:05–01:06 | the work commit `efd4e7e`, clean |
| `tests/make_test_*_at_efd4e7e.txt` | the work commit `efd4e7e`, clean |
| §1 | images built at `efd4e7e`; compose YAML from `efd4e7e` with the rig's deploy directory as the project directory |

## Timeline (UTC, 2026-11-04)

| event | time |
|---|---|
| the new tests on `main`'s code | 00:29:23, 00:29:54 |
| the signal probes | 00:31–00:32 |
| the fence probes on rmw's localhost config. The first attempt put the client in another namespace and never reached its goal; it was rerun with the client in the node's namespace, the file overwritten | 00:36–00:37 |
| the work commit `efd4e7e` | 00:38:06 |
| the hold opened; `make images` | 00:41:13; 00:41:40–00:58:37 |
| the bridge up on the sim host | 00:45:27 |
| the lease probes on both sides | 00:46–00:47 |
| the five lane services up; the warm-up spin complete | 00:59:18; 01:00:53 |
| the submit; the restart (the zero point, read just before the command); 30 s observed | 01:01:14.266; 01:01:24.877; to 01:01:56 |
| `compose down`; the bridge stopped | 01:02:39–01:02:43; 01:02:59–01:03:00 |
| the tags restored, `strafer_base` recreated (created, not started); the host released | 01:03:08; 01:03:19 |
| the one-sided lease probes; the as-deployed probes | 01:05; 01:06 |
| `make test-ros`, `make test-autonomy` at `efd4e7e` | 01:09 |
