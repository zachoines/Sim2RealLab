# Give the autonomy container a 3 s DDS participant lease

**Status:** Shipped 2026-11-08 in `8530e0e` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/266

**Type:** reliability (a crashed executor's policy goal drives on for about 10 s)
**Owner:** Jetson (`strafer_bringup` config, `deploy/`)
**Priority:** P1 for goal-b. On the real robot, about 10 s at the deployed speed cap is a long
unattended drive after an executor crash.
**Estimate:** S
**Branch:** `task/autonomy-cyclone-lease`

## Story

As **the maintainer running a policy goal on the real robot**, I want **a crashed or SIGKILLed
executor's goal to end within about 3 s**, so that **the robot does not keep following the policy
for 10 s after the process that commanded it is gone.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/deploy-env-config.md](../context/deploy-env-config.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

Read at `main` `9d970718`.

- **The second fence waits for the client's lease.** The inference node aborts a policy goal, with
  its stop, once the action's feedback topic has lost the subscriber it had
  ([`executor-exit-cancels-active-goal`](executor-exit-cancels-active-goal.md)).
  A SIGKILLed or crashed participant sends no goodbye, so it leaves the graph only when its DDS
  participant lease lapses.
  - The client's own lease sets the latency, at the lease less about 0.15 s
    ([`executor-exit-restart-2026-11-04`](../../measurements/executor-exit-restart-2026-11-04/README.md)
    §3).
  - Neither deployed Cyclone config sets a lease, so Cyclone's 10 s default applies: 9.81 s with
    the tracked `cyclonedds.xml` on both sides, 9.86 s with the rig's direct-link config.
  - A 3 s `Domain/Discovery/LeaseDuration` in the client's XML alone gave 2.85 s.
- **The decision.** The maintainer chose the narrowest of PR #264's three options: a lease in the
  autonomy container's Cyclone XML alone, 3 s, with the other containers' participants on the
  default.
- **Every CPU service mounts one file.**
  - The `*cpu-svc` anchor mounts `../strafer_bringup/config/cyclonedds.xml` at
    `/opt/strafer/config/cyclonedds.xml` (`docker-compose.yml`:62-63).
  - `autonomy` inherits the anchor and has no `volumes:` of its own (:133-137).
  - `CYCLONEDDS_URI` is one literal in the `x-dds-env` anchor (:47). `check_env_sync.py` pins it
    (:59, :146-147) and checks each Cyclone mount's target against it (:148-150).
  - That check finds mounts with the regex `cyclonedds\.xml:(/\S+?):ro` (:142), which does not
    match a file of another name.
- **The other lane.** `docker-compose.sim.yml` runs the hardware-free stack, executor included, in
  one container on the tracked file, and launches no inference node, so it has no fence.
- **Host-local overrides.** The untracked `deploy/docker-compose.override.yml` that Compose loads
  for a bare `docker compose up` can mount its own Cyclone XML at the same container path. An
  override's mount replaces a base mount with the same target, so on such a host `autonomy` gets
  the override's file.

## Acceptance criteria

- [x] **A Cyclone file for `autonomy` alone.** `strafer_bringup/config/cyclonedds.autonomy.xml`
      carries `cyclonedds.xml`'s `Internal` settings unchanged, plus
      `Domain/Discovery/LeaseDuration` at 3 s. Its header says what it is, why `autonomy` alone
      carries it, and that it tracks `cyclonedds.xml`. `cyclonedds.xml`'s header names it.
  - *2026-11-08: in `8530e0e`, with the lease written `3 s`, the text the record's probe used. The
    header also says that every process in the container carries the lease, `goal_projection`
    included. In `cyclonedds.xml`'s header, the `--` of `sudo sysctl --system` is not allowed
    inside an XML comment, so standard parsers rejected the file although Cyclone loaded it. The
    command became `sudo sysctl -p /etc/sysctl.d/99-cyclonedds-rmem.conf`.*
- [x] **`autonomy` mounts it** at the same container path, with the URI, the env files and the
      images unchanged.
  - *2026-11-08: `autonomy` has its own `volumes:` re-listing the anchor's one mount with the
    source swapped. `docker compose config` renders it read-only at
    `/opt/strafer/config/cyclonedds.xml`.*
- [x] **`check_env_sync.py` checks the new mount's target** against the URI, and `make env-check`
      passes.
  - *2026-11-08: the regex matches `cyclonedds(?:\.\w+)?\.xml:`. With the `autonomy` mount's target
    changed to a wrong path in a scratch copy, the check exits 1, and `main`'s regex exits 0.*
- [x] **A contract test** beside `test_teleop.py`:
  - the autonomy XML parses, and its element tree is `cyclonedds.xml`'s plus exactly one
    `Domain/Discovery/LeaseDuration` at 3 s;
  - `cyclonedds.xml` sets no `LeaseDuration`;
  - in the tracked compose files, `autonomy` mounts the autonomy XML at the URI's path, no other
    service mounts it, and every other Cyclone mount is the tracked file.

  The PR says which assertions fail on `main`, and why.
  - *2026-11-08: `test_autonomy_cyclone_lease.py`, 15 tests. On `main`, 5 fail:*
    - *two with `FileNotFoundError`, since there is no autonomy XML;*
    - *two with `ParseError` at line 16, column 77 of `cyclonedds.xml`;*
    - *one with `AssertionError`, since `autonomy` mounts `cyclonedds.xml`.*
- [x] **The file takes effect.** Cyclone accepts it, and the time from a SIGKILL of a client
      carrying it to the fence's abort is measured, or stated as unmeasured.
  - *2026-11-08: Cyclone's config trace shows `LeaseDuration` 3 s from the file, and 10 s from
    `cyclonedds.xml`. In the record's split probe, run on loopback in a container with no network,
    a SIGKILLed client carrying it was aborted 2.86 s later, against 9.82 s with `cyclonedds.xml`.
    On the same loopback, an idle participant carrying it stayed counted for 24.7 s. A 5 s SIGSTOP
    dropped it 2.33 s in, and it was counted again 0.03 s after SIGCONT.*
- [x] **Docs.** The deploy README's stop-signal note gives the executor's fence latency, about 3 s.
      `source/strafer_autonomy/README.md` changes only if it states that latency. Nothing in source
      or config names a record or a date.
  - *2026-11-08: the stop-signal note and the DDS line. `source/strafer_autonomy/README.md` states
    no latency and is unchanged.*
- [x] **A dated note** on the lease line of
      [`executor-exit-cancels-active-goal`](executor-exit-cancels-active-goal.md),
      naming the decision and this PR.
- [x] **The PR's Scope states the unmeasured cost.** A 3 s lease drops a live `autonomy` participant
      that falls silent for 3 s, with its data connections. The PR says where that would show, and
      that nothing ran on the robot.
  - *2026-11-08: it also says that every process in `autonomy` carries the lease, and that a
    host-local override replacing the mount's source removes it.*
- [x] No regression: `make test-ros` passes.
  - *2026-11-08: 1015 passed, 14 skipped, 1 xfailed at `8530e0e`.*

## Out of scope

- **A shorter lease in the shared configs**, which would apply to every participant on the domain.
- **Measuring the cost on a live participant**, under CPU starvation or a congested link.
- **The one-container sim lane** (`docker-compose.sim.yml`), which runs no inference node.
- **Host-local Cyclone overrides.** A host that mounts its own file for `autonomy` carries the lease
  only if that file sets it.
- **A run outside the containers.** `env_autonomy.env` points every node at `cyclonedds.xml`.
- **What masks the fence:** any other subscriber to the action's feedback topic.
