# Give the autonomy container a 3 s DDS participant lease

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

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/deploy-env-config.md](../../context/deploy-env-config.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

Read at `main` `9d970718`.

- **The second fence waits for the client's lease.** The inference node aborts a policy goal, with
  its stop, once the action's feedback topic has lost the subscriber it had
  ([`executor-exit-cancels-active-goal`](../../completed/executor-exit-cancels-active-goal.md)).
  A SIGKILLed or crashed participant sends no goodbye, so it leaves the graph only when its DDS
  participant lease lapses.
  - The client's own lease sets the latency, at the lease less about 0.15 s
    ([`executor-exit-restart-2026-11-04`](../../../measurements/executor-exit-restart-2026-11-04/README.md)
    §3).
  - Neither deployed Cyclone config sets a lease, so Cyclone's 10 s default applies: 9.81 s with
    the tracked `cyclonedds.xml` on both sides, 9.86 s with the rig's direct-link config.
  - A 3 s `Domain/Discovery/LeaseDuration` in the client's XML alone gave 2.85 s.
- **The decision.** The maintainer chose the narrowest of PR #264's three options: a lease in the
  autonomy container's Cyclone XML alone, 3 s, with every other participant on the default.
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

- [ ] **A Cyclone file for `autonomy` alone.** `strafer_bringup/config/cyclonedds.autonomy.xml`
      carries `cyclonedds.xml`'s `Internal` settings unchanged, plus
      `Domain/Discovery/LeaseDuration` at 3 s. Its header says what it is, why `autonomy` alone
      carries it, and that it tracks `cyclonedds.xml`. `cyclonedds.xml`'s header names it.
- [ ] **`autonomy` mounts it** at the same container path, with the URI, the env files and the
      images unchanged.
- [ ] **`check_env_sync.py` checks the new mount's target** against the URI, and `make env-check`
      passes.
- [ ] **A contract test** beside `test_teleop.py`:
  - the autonomy XML parses, and its element tree is `cyclonedds.xml`'s plus exactly one
    `Domain/Discovery/LeaseDuration` at 3 s;
  - `cyclonedds.xml` sets no `LeaseDuration`;
  - in the tracked compose files, `autonomy` mounts the autonomy XML at the URI's path, no other
    service mounts it, and every other Cyclone mount is the tracked file.

  The PR says which assertions fail on `main`, and why.
- [ ] **The file takes effect.** Cyclone accepts it, and the time from a SIGKILL of a client
      carrying it to the fence's abort is measured, or stated as unmeasured.
- [ ] **Docs.** The deploy README's stop-signal note gives the executor's fence latency, about 3 s.
      `source/strafer_autonomy/README.md` changes only if it states that latency. Nothing in source
      or config names a record or a date.
- [ ] **A dated note** on the lease line of
      [`executor-exit-cancels-active-goal`](../../completed/executor-exit-cancels-active-goal.md),
      naming the decision and this PR.
- [ ] **The PR's Scope states the unmeasured cost.** A 3 s lease drops a live `autonomy` participant
      that falls silent for 3 s, with its data connections. The PR says where that would show, and
      that nothing ran on the robot.
- [ ] No regression: `make test-ros` passes.

## Out of scope

- **A shorter lease in the shared configs**, which would apply to every participant on the domain.
- **Measuring the cost on a live participant**, under CPU starvation or a congested link.
- **The one-container sim lane** (`docker-compose.sim.yml`), which runs no inference node.
- **Host-local Cyclone overrides.** A host that mounts its own file for `autonomy` carries the lease
  only if that file sets it.
- **A run outside the containers.** `env_autonomy.env` points every node at `cyclonedds.xml`.
- **What masks the fence:** any other subscriber to the action's feedback topic.
