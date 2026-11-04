# The executor's exit cancels its policy goal, 2026-11-04

[`executor-exit-cancels-active-goal`](../../tasks/completed/executor-exit-cancels-active-goal.md)
gives the executor its own SIGINT and SIGTERM handling, so that its exit path cancels the active
mission and publishes a zero `/cmd_vel` while the ROS context is still live. It also gives the
inference node a client-liveness check that ends a goal whose client is gone. This record checks
both at the work commit `efd4e7e`:
- **on the sim-bridge lane** with a policy backend, `docker compose restart autonomy` during a
  policy goal;
- **in the image**, the executor's exit under SIGINT and SIGTERM, and the inference node's loop
  after its client process is SIGKILLed.

It makes **no hardware claim**. No camera, no RoboClaw and no robot motion were involved. The robot
on the lane was the sim's.

**Result.**
- **The restart ended the goal with a cancel and a stop** (§1). Times are wall time on the robot
  host, relative to 01:01:24.877 UTC, the instant read just before `docker compose restart
  autonomy` ran (§1):

  | event | time |
  |---|---|
  | the executor logs "Shutting down autonomy executor." | +0.358 s |
  | the last non-zero `/cmd_vel` | +0.277 s |
  | **the stop: the first zero `/cmd_vel` after it** | **+0.382 s** |
  | the inference node logs "navigate_to_pose canceled by client." | +0.403 s |
  | the last `/cmd_vel`, a zero; nothing on `/cmd_vel` in the 30 s after | +0.403 s |
  | the restarted executor is ready | +3.326 s |

  The bridge ran at about 0.13 of real time, so the 0.382 s is about 0.05 s of sim time.
- **The exit path runs on a live context** (§2). In the image, the fixed executor destroys its
  server with `rclpy.ok()` true, 23 ms after SIGINT and 28 ms after SIGTERM, and exits 0. `main`'s
  executor reaches its destroy with `rclpy.ok()` already false.
- **The second fence's latency is the dead client's DDS lease** (§3). As deployed, a SIGKILLed
  client's goal was aborted with a stop 9.81 s (the tracked `cyclonedds.xml`) and 9.86 s (the
  rig's direct-link config) after the kill. Those configs set no lease, so Cyclone's default
  applies. A 3 s lease in the client's XML alone brought it to 2.85 s.
- **Tests** (§4). The new tests fail on `main` and pass at `efd4e7e`. `make test-ros` passes 990
  and `make test-autonomy` passes 604.

## 1. The bridge-lane restart

The five lane services (`inference`, `slam`, `navigation`, `sim-perception` and `autonomy`) ran on
images built at `efd4e7e52a1f` (`strafer-cpu` `09206258885a`, `strafer-gpu` `0fdfa153c046`).
- **Compose.** This branch's `docker-compose.yml` and `docker-compose.override.sim-bridge.yml`,
  with the rig's host-local overrides. `--project-directory` was the rig's deploy directory, so
  the rig's `.env`, models, policy backend (`hybrid_nav2_strafer`, `DEPTH_SUBGOAL`) and Cyclone
  pins applied unchanged. Every service rendered `stop_signal: SIGINT` (`lane/up_checks.txt`).
  SLAM ran on its own token, `ee1`.
- **The bridge** was the inside-goal lane's room F1 on the sim host, spawning at world
  (−1.65, −0.55).
- **The goal.** `restart_check.sh` waited for 5 s of quiet `/cmd_vel` after the warm-up spin. It
  read `map→base_link` (yaw −36.0°, inside the −50°…+75° range with at least 2 m clear) and
  submitted "move 1.800 meters forward and 0.000 meters left" with `strafer-autonomy-cli` inside
  `strafer_autonomy`, as `make submit-deploy` does. The executor sent a hybrid goal. The inference
  node's active-goal message came 7.0 s after the submit, and the first non-zero `/cmd_vel` 0.43 s
  after that.
- **The restart.** It went in 3.18 s into the drive and returned 1.26 s later.
  - **The zero point** is `time.time()` read by `restart_check.sh` just before the command:
    1793754084.877, that is 01:01:24.877 UTC (`lane/restart_t.txt`, first field).
  - **The two stamps after it** are not separate restart times. The UTC stamp beside it
    (01:01:24.885) is a `date` run 8 ms later on the same line. The `host_hold.log` line
    (01:01:24.889) was written next, and then `docker compose restart autonomy` ran.
  - So the command started at most about 12 ms after the zero point, and every offset
    overstates the delay from the command's start by at most that much.
- **The timing.** `cmdvel_logger.py` ran in `strafer_navigation`. It subscribes to no action topic,
  so it never counts as an action client, and it stamped each `/cmd_vel` with the receiving
  wall time. The container logs carry Docker's timestamps.

The log order matches the exit path. The cancel goes out and its response returns, then the
executor publishes its zero (+0.382 s). The executor then destroys its server and shuts rclpy down
(+0.393 s), and the inference node's loop sees the cancel, logs it and publishes its own stop
(+0.403 s). Three lines in the executor's log at the exit are not part of the fix:
- the ROS client's spin thread raising `ExternalShutdownException` at `rclpy.try_shutdown()`, as
  on `main` (§2);
- "Failed to send result response", the mission's result to the CLI;
- the CLI's exit 137 (`lane/cli.out`). It ran inside the restarted container, so it ended with it.

**The lane's own compose chain sets no stop signal.** Rendered from the rig checkout that the
inside-goal lane runs compose from, a branch that predates
[`compose-stop-signal-sigint`](../../tasks/completed/compose-stop-signal-sigint.md), all five
services read `stop_signal` None (`lane/stop_signal_by_chain.txt`). A stop or restart from that
checkout sends SIGTERM, which launch ignores, so Docker kills the services at 10 s and the
executor's handler never runs. This check used `main`'s compose YAML for that reason.

## 2. The executor's signal path, in the image

`probes/signal/probe.sh` runs the executor's `main()` in `strafer-cpu:humble` with `--network none`,
against stub planner and VLM health endpoints on loopback. It sends the signal one second after the
ready line. `AutonomyCommandServer.destroy` is wrapped to print `rclpy.ok()` and the time.

| executor | signal | `rclpy.ok()` at the destroy | signal → destroy | signal → exit | exit |
|---|---|---|---|---|---|
| `efd4e7e` (the tree mounted) | SIGINT | True | 23 ms | 0.34 s | 0 |
| `efd4e7e` (the tree mounted) | SIGTERM | True | 28 ms | 0.36 s | 0 |
| `main` (the image's own) | SIGINT | **False** | 8 ms | 0.33 s | 0 |

Both executors print the spin thread's `ExternalShutdownException` at exit. On `main`, rclpy's own
handler shuts the context down at the signal, so the destroy, and any cancel there, runs on a dead
context.

## 3. The second fence and the lease

`probes/fence/` runs the inference node's execute loop for a mock goal, in one process, against an
action client in a second process. It waits until the node counts the client's feedback
subscription, starts the loop, SIGKILLs the client and times the abort. Each side can take its own
Cyclone XML. Every run ended with one abort and a zero as the last command.

| client's Cyclone XML | node's Cyclone XML | `ROS_LOCALHOST_ONLY` | SIGKILL → abort |
|---|---|---|---|
| **tracked `cyclonedds.xml`** | **the same** | unset, as deployed | **9.81 s** |
| **the rig's direct-link config** | **the same** | unset, as deployed | **9.86 s** |
| direct-link + a 3 s lease | direct-link | unset | 2.85 s |
| a 5 s lease | a 5 s lease | 1 | 4.84 s |
| a 10 s lease | a 10 s lease | 1 | 9.85 s |
| a 5 s lease | none | 1 | 4.85 s |
| a 3 s lease | none | 1 | 2.85 s |
| none | a 5 s lease | 1 | 9.82 s |
| none | a 3 s lease | 1 | 9.82 s |
| none | none | 1 | 19.75 s, 19.75 s |

- **The mechanism.** The node's loop checks every 50 ms whether the action's feedback topic still
  has the subscriber it had. A SIGKILLed participant sends no goodbye, so it leaves the graph only
  when its participant lease lapses (Cyclone's `Domain/Discovery/LeaseDuration`). The client's
  own lease sets the latency, at the lease less about 0.15 s.
- **As deployed** neither config sets a lease, so the fence ends a crashed executor's goal after
  about 10 s.
- **What masks it.** The fence counts every subscriber to `navigate_to_pose/_action/feedback`, not
  the goal's own client. So any other subscriber keeps a dead client's goal alive until it too is
  gone: a second action client, a `ros2 topic echo` or a logger on that topic, or a Foxglove client
  subscribed to it.
  - **In the tree today** the executor's one client is the only subscriber: `ros_client.py`'s
    `_strafer_direct_client`, shared by `strafer_direct` and hybrid goals. The tree's other action
    clients are Nav2's (`navigate_to_pose`, `compute_path_to_pose`) and the `ExecuteMission`
    client, and `cmdvel_logger.py` subscribes to no action topic.
  - **The viewer** sets no `include_hidden`. Whether foxglove_bridge offers the hidden `_action`
    topics is therefore its default's; its own launch file declares false.
  - **The sim-bridge rig's mission observer**, outside the tree, does subscribe to the feedback
    topic, so the fence is masked while it runs.
- **The name.**
  - The deployed node runs in the `strafer_inference` namespace (`inference.launch.py`, included
    by `inference_policy.launch.py`). The server's `navigate_to_pose` and the count's
    `navigate_to_pose/_action/feedback` both expand under it, to
    `/strafer_inference/navigate_to_pose`, the name the executor's client uses.
  - The launch remaps only `/strafer/cmd_vel`, so the deployed node counts the topic the client
    subscribes to.
  - The lane check exercised the cancel path. The fence itself was measured in the probe, with
    both sides in the root namespace.
- **The no-XML rows on both sides** come from rmw's built-in localhost configuration, which is not
  the deployed case. Why that pairing gives about twice the default was not established.
- **A clean exit** deletes the participant at once, so the fence fires on its next 50 ms poll.

## 4. Tests

- `TestShutdown` in `tests/test_executor_main.py` has three cases: with a mission active, the exit
  path cancels the active navigation and publishes a zero `/cmd_vel` before the server is
  destroyed; with none active it cancels nothing; and a raising cancel still lets the destroy run.
  On `main` all three fail, because no `shutdown` exists there (`tests/test_on_main_autonomy.txt`).
- `test_a_goal_whose_client_is_gone_ends_with_a_stop` in `test_inference_runtime.py` starts a goal
  with an in-process action client, destroys the client and requires the goal to abort with a stop
  within 1 s. On `main` the goal was still driving 5 s after its client went away
  (`tests/test_on_main_ros.txt`).
- At `efd4e7e`: `make test-ros` 990 passed, 11 skipped, 1 xfailed; `make test-autonomy` 604
  passed, 83 skipped. The container lane's known collection error on
  `test_sim_in_the_loop_runtime_env.py` (`isaaclab_tasks` not importable) is ignored, as on `main`
  (`tests/make_test_autonomy_plain_at_efd4e7e.txt`).

## What this does not show

- **No real robot.** `base` did not run, so the driver's view of the zero twist is not here.
- **One stop path on the lane.** `docker compose stop` and `down`, and a Ctrl-C to a foreground
  `docker compose up`, deliver the same SIGINT through `stop_signal`. They were not run separately.
  The image probes cover SIGINT and SIGTERM to the executor itself.
- **The crash on the lane.** The fence after a SIGKILL was measured only in the probe, not on the
  lane.
- **The fence under another subscriber.** A second subscriber to the feedback topic masks the fence
  (§3). It was not run with one.
- **A shorter lease's cost.** A shorter lease drops a live participant that falls silent sooner,
  for example under CPU starvation or a congested link. A dropped participant also loses its data
  connections. That is not measured here.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `executor-exit-restart-2026-11-04/record-files/` |
| deposit commit | `5dc7e09ab1e7e3b4d2d63fb28fe959f0d13a9794` |

The deposit holds:
- `host_hold.log`, session strafer-nx's hold on the robot host, and `dgx/host_hold.log`, the sim
  host's bridge launch and stop;
- `check/`, the check's scripts, and `lane/`, its compose config, the five containers' logs, the
  `/cmd_vel` ride-along and `summary.txt`;
- `build/`, the image build log and the restored tags;
- `tests/`, the runs on `main` and at `efd4e7e`;
- `probes/signal/` and `probes/fence/`, the probes and their outputs;
- `DEPOSIT.md`, with the sha256 of every file. The rig's network addresses are redacted to
  `<addr>` in two files.

Paths in this record resolve in that directory. Verify the digests with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/executor-exit-restart-2026-11-04/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

Hosts, trees and times are in [`provenance.md`](provenance.md).
