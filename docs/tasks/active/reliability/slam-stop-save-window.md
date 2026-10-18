# Let rtabmap finish its save when slam stops

**Type:** bug (launch ends rtabmap's close-time save within 5–10 s, whatever slam's grace period)
**Owner:** Jetson (`deploy/`, `strafer_slam`)
**Priority:** P2. Before [`compose-stop-signal-sigint`](../../completed/compose-stop-signal-sigint.md)
a stop started no save at all. Now the save starts, and launch ends one that runs too long. How
long a real map's save takes is unmeasured. The 1.25 GB database in
[`enriched-lane-rig-stability`](enriched-lane-rig-stability.md)'s mode 1 may be in the size class
where that matters, and so, over time, may the robot's one persistent map.
**Estimate:** S
**Branch:** `task/slam-stop-save-window`

## Story

As **whoever stops or restarts `slam` on a large map**, I want **rtabmap's close-time save to
finish inside slam's grace period**, so that **the next load finds a complete database: no repair
warning and no `addLink()` abort.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`compose-stop-signal-2026-10-17`](../../../measurements/compose-stop-signal-2026-10-17/README.md)
  §1-§3: the stop signal, launch's escalation, and the probe that raises it.
- [`enriched-lane-rig-stability`](enriched-lane-rig-stability.md), mode 1 and its open
  runtime-verification box.

## Context

Read and probed on 2026-10-17 in `strafer-cpu:humble` (launch 1.0.14), compose at `8230e29`.

- **The stop now reaches rtabmap.** `stop_signal: SIGINT` makes launch take its Ctrl-C path, and
  rtabmap logs "Saving database/long-term memory..." on a stop. Before, no save line ever appeared:
  launch catches no SIGTERM, so nothing reached rtabmap.
- **Launch bounds the save at 5 to 10 s.**
  - Each process still running `sigterm_timeout` seconds after the SIGINT gets SIGTERM. One still
    running `sigterm_timeout + sigkill_timeout` seconds after it gets SIGKILL. Both are launch
    configurations defaulting to 5 (`launch/actions/execute_local.py`:90-93, :443-479).
  - slam's `stop_grace_period: 180s` bounds only Docker's own SIGKILL.
  - In the probe, a child whose teardown takes 20 s, and which ignores SIGTERM, was SIGKILLed at
    10.0 s, and the container still exited 0. With the launch argument `sigterm_timeout:=60` it
    finished in 20.0 s.
  - rtabmap's rclcpp handles SIGTERM as it handles SIGINT (`signal_handler(SIGINT/SIGTERM)`).
    Whether a save in progress survives that 5 s SIGTERM is unmeasured.
- **The timed save was trivial.**
  - The record timed a new, empty database. It was last written 53 ms after launch passed on
    the SIGINT, and rtabmap exited 0.27 s after it.
  - rtabmap's "Saving database/long-term memory..." lines are plain console output that arrived
    after that last write, likely flushed at exit, so their arrival does not time the save.
  - A real map's save time is unmeasured. Mode 1 holds that it "scales with the working set being
    serialised, not with the 1.25 GB file itself", and that a save of that map "does not complete
    in 10 s". That second sentence was never measured: the stop never reached rtabmap then.
- **Where slam runs:**
  - `deploy/docker-compose.yml`'s `slam`: `ros2 launch strafer_slam slam.launch.py`, 180 s grace;
  - `docker-compose.override.sim-bridge.yml` replaces that command with `use_sim_time:=true` added;
  - `docker-compose.sim.yml`'s `strafer-sim` includes slam through
    `bringup_sim_in_the_loop.launch.py`, on Docker's 10 s default grace;
  - the bare-metal bringup launch files include `slam.launch.py`.
- **Two levers.**
  - Launch arguments on each compose command. They apply to every process in that launch.
  - `sigterm_timeout=` / `sigkill_timeout=` on the rtabmap `Node` in `slam.launch.py`. They apply
    to rtabmap in every caller.

  Either way, the sum must stay under the container's grace, or Docker's SIGKILL comes first.

## Acceptance criteria

- [ ] **rtabmap's save time on a real-size map**, measured with `stop_signal: SIGINT` in place,
      on a database of the enriched lane's size or on the robot's persistent map:
      - time it from launch's SIGINT to the database's last write and to rtabmap's exit, not to
        the console line's arrival;
      - with launch's timeouts raised for the run, so the save is not cut;
      - record whether a save survives launch's 5 s SIGTERM;
      - record the database size and the working-memory size.
- [ ] **The lever chosen and set**, with the reason, so that rtabmap's save is bounded by a
      window larger than the measured save and smaller than the container's grace. This covers
      `slam` and, if `strafer-sim` stays in use, its grace too.
- [ ] **A test pins the configuration.** For example, `test_bringup.py` reads the timeouts that
      reach rtabmap and checks them against slam's `stop_grace_period`.
- [ ] **A stop and start on the large database** shows the save's "done!" line on the stop and no
      `loadDataFromDb()` repair warning on the next load. This is also
      `enriched-lane-rig-stability`'s runtime-verification box.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` and `make env-check` pass.

## Investigation pointers

- `source/strafer_ros/strafer_slam/launch/slam.launch.py`: the rtabmap `Node`.
- `source/strafer_ros/deploy/docker-compose.yml` (`slam`), `docker-compose.override.sim-bridge.yml`
  (`slam`'s command), `docker-compose.sim.yml` (`strafer-sim`).
- In the image: `launch/actions/execute_local.py` (the timers, :443-479).

## Out of scope

- dockerd's `shutdown-timeout` on a daemon stop or a reboot. That residual is in
  `enriched-lane-rig-stability`.
- rtabmap's `addLink()` assertion upstream.
