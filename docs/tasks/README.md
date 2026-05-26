# `docs/tasks/` — work queue + authoring guide

This directory holds **one-shot, agent-pickup-able work briefs** in a
Jira-style format. Each `.md` here describes a single discrete piece of
work that a fresh agent (DGX, Jetson, or either) can pick up and
execute against without needing the operator to also write a long
preamble each time.

Briefs are persistent across sessions. When you ship a brief, the
housekeeping (stamp + move to [`completed/`](completed/) + update
[`BOARD.md`](BOARD.md)) happens **inside the same PR as the work,
before merging** — the exact sequence is documented under
[Shipping a brief: order of operations](#shipping-a-brief-order-of-operations)
below. Doing it after merge is the failure mode and has bitten us
before; the rule is now strict.

The active queue stays scannable while shipped briefs remain
discoverable as a record of what we asked for, the acceptance
criteria we held the change to, and which follow-ups it spawned.
Git history records the *what changed*; the brief records the *what
we set out to do* — both are useful, and they're different artifacts.

**If you've landed here as a fresh agent looking for work, open
[`BOARD.md`](BOARD.md).** It's the glanceable index of the queue,
sorted by priority + lane + estimate, with explicit "in flight" and
"blocked" sections so you don't pick something already in review or
gated on unshipped work.

If you're writing a new brief, read on.

---

## Where the reusable preamble lives

Every brief used to start with ~50 lines of duplicated preamble:
"who you are," "what hosts," "ownership boundaries," "branch state,"
etc. That preamble has been extracted into reusable modules under
[`context/`](context/). Briefs reference modules by link rather than
re-stating their facts.

See [`context/README.md`](context/README.md) for the context-system
contract (when to add a module, how to maintain it, the acceptance
bullet that prevents context drift).

---

## File naming

Lowercase kebab-case, descriptive of the work, no leading numbers:

- `real-d555-depth-range-survey.md`
- `async-camera-publishers.md`
- `jetson-headless-viewer.md`
- `integration-prompts-refresh.md`

Avoid prefixes like `001-`, dates, or owner initials. The `Owner` /
`Priority` / `Estimate` fields in the frontmatter convey ordering and
ownership; the filename should still read sensibly when you scan
the active queue.

When a brief lives inside an epic subdir (see
[`## Directory layout`](#directory-layout) below), the filename may
elide a prefix that the subdir already conveys —
`harness/teleop-driver.md` rather than `harness/harness-teleop-driver.md`.
This is opportunistic: shorten when the result reads naturally,
otherwise keep the longer form.

---

## Directory layout

`docs/tasks/` has four siblings:

| Path | What lives here |
|---|---|
| [`active/<epic>/`](active/) | Pickable briefs, grouped by feature epic |
| top-level `active/<brief>.md` (rare) | Meta exception: a brief that defines the layout itself sits at top-level rather than inside an epic. The current pattern was set by [`completed/task-board-epic-structure.md`](completed/task-board-epic-structure.md). Use sparingly. |
| [`parked/<epic>/`](parked/) | Filed-on-trigger or blocked-on-deps briefs, mirroring `active/`'s subdir structure |
| [`completed/`](completed/) | Historical record, kept **flat** (no epic subdirs) — completed briefs are browsed by date / search, not by feature area |
| [`context/`](context/) | Cross-epic context modules referenced by brief `## Context bundle` sections; flat by design (modules don't belong to epics) |

### Epic dirs

The nine epics are a fixed small set, decided per
[`context/conventions.md`'s task-board section](context/conventions.md#task-board-structure).
Adding a new epic is a convention change, not an ad-hoc decision —
file a brief that amends this layout. Do **not** invent a `misc/`
bucket.

Current epics:

- `multi-room/` — cross-room navigation
- `trained-policy/` — RL training + Jetson policy deployment
- `harness/` — training-data pipeline
- `clip-validation/` — mid-mission validation via CLIP
- `sim-performance/` — bridge throughput, renderer
- `reliability/` — nav + executor bugs + autonomy refactors
- `tooling/` — testing, CI, workstation bringup
- `experimental/` — long-horizon research bets
- `investigations/` — one-shot measurements that spawn follow-ups

### Parked sibling

`parked/<epic>/<brief>.md` is the home for briefs that are filed
but **not pickable right now**. Two common reasons:

- **Filed-on-trigger.** Pickup gated on a condition the operator
  decides (e.g., "pick up only when the v1 watchdog produces
  real-world false positives").
- **Blocked-on-deps.** Pickup gated on another brief shipping.

Un-park is one `git mv`:

```
git mv parked/<epic>/<brief>.md active/<epic>/<brief>.md
```

This goes in the PR that picks the brief up; update the brief's
row in [`BOARD.md`](BOARD.md) in the same commit (state: `parked`
→ `in flight`).

### Branch naming

`task/<basename>` — the epic subdir does **not** leak into the
branch name. `active/harness/mission-generator.md` → branch
`task/mission-generator`. The path organizes the filesystem; the
branch organizes the worktree.

---

## Brief format

```markdown
# <One-line title — verb-leading where it makes sense>

**Type:** task / bug / investigation / refactor / docs
**Owner:** DGX agent / Jetson agent / either
**Priority:** P0 / P1 / P2
**Estimate:** S / M / L  (rough effort; ~hours / ~day / ~multi-day)
**Branch:** task/<brief-slug>  (= filename minus .md, prefixed with
                                "task/"; see ../context/branching-and-prs.md)

## Story

As a **<role>**, I want **<action>**, so that **<outcome>**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
  *(omit modules that are not relevant to this task; branching-and-prs
   is universally relevant)*

## Context

<background, code pointers, hypotheses, and the symptom that
triggered this task — anything specific to THIS task that the
context modules don't already cover. Keep stable facts in modules;
keep this section about the task itself.>

## Acceptance criteria

- [ ] <criterion 1 — concrete, falsifiable>
- [ ] <criterion 2>
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports
      (call out the relevant smoke test).

## Investigation pointers

<file paths with line numbers, runtime commands, related task
briefs, prior art. Link directly with `path/to/file.py:42`.>

## Out of scope

<things explicitly deferred. Avoids agents drifting into adjacent
work and producing oversize PRs.>
```

### Required vs optional sections

- **Required:** Title, the five frontmatter fields (Type, Owner,
  Priority, Estimate, Branch), Story, Mission / Context (one or the
  other), Acceptance criteria, Out of scope.
- **Recommended:** Context bundle (skip only for trivial,
  self-contained tasks).
- **Optional:** Investigation pointers (skip when there's nothing
  to point at). Open questions, risks, ranked-hypotheses tables —
  add when the task warrants.

### The acceptance bullet that keeps context current

Every brief must include this bullet (or an equivalent):

```markdown
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
```

This is the forcing function that prevents documentation drift. The
agent that ships the change is the one closest to the new truth —
easier to update one fact in one place than to ripple it through
half a dozen briefs / READMEs / runbooks later.

The original narrower form (context modules only) appears in older
briefs and is not invalidated retroactively; new briefs use the
broader form so package READMEs, the top-level `Readme.md`, and the
`docs/` guides stay aligned with the system they describe.

---

## How to compose context for a new task

1. **Identify which modules cover preamble you'd otherwise duplicate.**
   Skim [`context/`](context/). If your brief would otherwise repeat
   "DGX is at 192.168.50.196 / conda env is env_isaaclab3 / branch
   off `main`" — that's a `context/repo-topology.md` link.

2. **Pull only what's relevant.** A pure-DGX bridge perf task probably
   needs `repo-topology` + `bridge-runtime-invariants`. A Jetson
   executor refactor probably needs `repo-topology` +
   `ownership-boundaries`. Don't bundle modules an agent doesn't
   need.

3. **If nothing in `context/` covers a recurring fact your brief
   needs, consider adding a module.** Tip: wait until the SAME fact
   would have appeared in 3+ briefs before promoting it. One-off
   facts belong in the task itself, not in shared context.

4. **Keep `## Context` (the task-specific section) about the task,
   not about the system.** "The bridge runs at decimation=1" is a
   system fact (context module). "On 2026-04-27 we observed
   timeouts at 90 s wall while sim ran at 3.6% RTF" is a task fact
   (this brief).

---

## How to write a good brief

A few rules of thumb that hold up in practice:

- **Lead with the symptom, not the diagnosis.** The agent should
  understand *what's wrong* before they see your guess at *why*.
  Diagnosis goes in the Context or Investigation Pointers section.
- **Make acceptance criteria falsifiable.** "Robot moves at usable
  speed" is wishful thinking. "Bridge `/clock` rate ≥ 8 Hz on the
  InfinigenPerception scene at decimation=1, measured by
  `ros2 topic hz`" is a check.
- **Quote line numbers when you have them.** `runtime_env.py:181-186`
  saves the agent a 30-second grep.
- **Out-of-scope is a feature.** It tells the agent "you might
  notice this, don't fix it here." Prevents PR-sprawl.
- **Don't over-spec the implementation.** "Use a Python rclpy
  thread mirroring `StraferAsyncPublisher`" is fine guidance;
  prescribing the line-by-line refactor isn't.
- **Keep cross-brief links resolvable.** When you reference a
  sibling brief, use its post-reorg path (`active/<epic>/<basename>.md`
  or `parked/<epic>/<basename>.md`) — not the legacy flat slug.
  Run `tools/check_brief_links.sh` from the repo root before
  pushing; it greps every markdown link under `docs/tasks/` and
  exits non-zero if any target doesn't resolve. Cheap to run
  locally; intended for pre-commit or CI.

### Anti-patterns

- **Status briefs.** "Investigation of why X is slow" with no
  Acceptance criteria. If there's no done-definition, it's not a
  task — it's a note. Put it in `docs/PERF_INVESTIGATION_*.md` or
  similar.
- **Multi-task briefs.** "Refactor A and also fix B and also
  document C" — split into three briefs. The Jira-style format is
  designed for one ticket = one PR.
- **Stale briefs.** When work ships, the housekeeping (stamp,
  move to [`completed/`](completed/), update [`BOARD.md`](BOARD.md))
  has to happen **inside the PR that ships the work, before
  merging.** See
  [Shipping a brief: order of operations](#shipping-a-brief-order-of-operations)
  below for the exact sequence. Living briefs in the top level that
  nobody owns become noise; an unstamped brief in `completed/` is
  ambiguous between "shipped" and "abandoned"; a board entry that
  points at a `completed/` path makes the index lie about its own
  contents. Doing the housekeeping after merge is the failure mode
  and has bitten us once already.

---

## Shipping a brief: order of operations

When you ship a brief, the housekeeping happens **inside the PR
that ships the work, before merging** — not after. Sequence on
your task branch:

1. **Land all work commits.** The actual code changes, tests,
   docs, etc. that satisfy the brief's acceptance criteria.

2. **As one of the last commits before requesting review, do the
   housekeeping in a single commit.** Three things together:

   a. Stamp the brief at the top, above the existing fields:

      ```markdown
      **Status:** Shipped <YYYY-MM-DD> in `<ship-commit>` (<host>).
      **PR:** https://github.com/<org>/<repo>/pull/<N>
      ```

      `<ship-commit>` is the *work commit* on this branch (the
      commit that lands the substantive change), not the merge
      commit — the merge commit doesn't exist yet. `<host>` is
      `Jetson` / `DGX` / `Either` per the brief's `Owner` field.

   b. `git mv docs/tasks/active/<epic>/<brief>.md
      docs/tasks/completed/<brief>.md`. `completed/` is flat — the
      epic subdir is dropped on ship. If the brief was parked when
      picked up, the source path is `docs/tasks/parked/<epic>/<brief>.md`
      (un-park to active first, then ship per the normal flow).

   c. Update [`BOARD.md`](BOARD.md):
      - Remove the brief's row from **In flight**. If In flight is
        now empty, leave a single `_None._` row.
      - Remove the brief's row from **By epic** and from **Ready
        to pick up**.
      - If the brief's validation surfaced follow-ups that you
        filed during this PR, ensure they're listed under
        **Ready to pick up** (or **Parked** if they have
        unshipped dependencies / filed-on-trigger semantics), and
        added under the right epic in **By epic**.

3. **Request review and merge.** The PR cannot merge without the
   housekeeping commit. Reviewers should treat its absence as a
   blocker.

### Why "in PR, before merge" is strict

The looser "do it after merge eventually" rule has dropped before:
[`completed/nav2-far-goal-staging.md`](completed/nav2-far-goal-staging.md)
shipped in PR #13 / 2026-04-29 but its brief sat in the active
queue until PR #15 caught it a week later. The active queue
silently misrepresented work-in-flight for that whole window. The
in-PR rule removes the discretion: there's no "later," there's
just "before merge."

The flipside is that the brief's `Status:` stamp uses the *ship
commit* on the branch, which exists at PR-creation time — not the
merge commit, which doesn't yet. That's a deliberate trade: the
ship commit is identifiable now, and `git log --oneline` /
`git show <ship-commit>` resolves the same actual change either
way.

### What "shipping a brief" means in this convention

Only one brief per PR is moved on ship — the brief whose work the
PR was opened to execute. Briefs *filed* during the same PR (as
follow-ups surfaced by the work) stay in active queue under
**Ready to pick up** — they're new work, not shipped work.

If a PR ships work that doesn't correspond to a brief (small
hotfixes, bumps, etc.), there's nothing to move; the BOARD.md
update is also unneeded. The rule applies only to the brief→PR
1:1 case the workflow is built around.

---

## How to kick off a task with a new agent

Briefs in this directory are **self-contained as far as task content
goes** — the context modules they reference cover all the preamble.
But a fresh agent does still need a thin launcher message giving them
four things the brief itself doesn't carry: their identity, a pointer
to which brief, workspace state, and stop conditions.

### Recommended templates

Two variants. Pick by whether you have a specific brief in mind for
this agent or you want them to choose from the queue.

**Variant A — specific brief assigned (most common):**

```
You are the [DGX | Jetson]-side coding agent. Your task brief is at:

    docs/tasks/active/<epic>/<brief-name>.md

Start by reading the brief and the context modules it links to from
its "## Context bundle" section. Workspace is already set up
(using [env_name] conda env; making sure to create your own branch per conventions) — no
setup commands needed.

When you're done — or if you hit a blocker — give me a short summary
of what landed plus any remaining open questions.
```

**Variant B — agent picks from the board:**

```
You are the [DGX | Jetson]-side coding agent. Open
docs/tasks/BOARD.md and pick the next item from "Ready to pick up"
that matches your lane and a [S | M | L]-sized session. Read the
chosen brief and the context modules it links to before starting.

Move the brief from "Ready to pick up" → "In flight" on
docs/tasks/BOARD.md in the commit that opens the PR (per the
maintenance contract at the bottom of that file). Workspace is
already set up (using [env_name] conda env; making sure to create
your own branch per conventions) — no setup commands needed.

When you're done — or if you hit a blocker — give me a short summary
of what landed plus any remaining open questions.
```

Variant A is the default; the brief is durable, the launcher is
throwaway. Use Variant B when the queue priority is clearer than the
specific work item, or when you want the agent to exercise judgment
on session-fit.

### What each line does

- **Identity** (`You are the [DGX | Jetson]-side coding agent`). The
  brief's `Owner:` field says this too, but front-loading it
  short-circuits any "wait, am I supposed to be touching this lane?"
  hesitation. The agent immediately knows which lane in
  [`context/ownership-boundaries.md`](context/ownership-boundaries.md)
  applies.
- **Pointer to the brief.** Agents can't psychically know what to
  read first. Always include the full path.
- **Workspace expectation.** Tells them not to spend cycles on
  `pip install -e`, `colcon build`, `git checkout`, etc. If the
  workspace actually isn't ready (stale branch, missing build, env
  not active), say so explicitly instead.
- **Permission.** Without it, agents default to feature-branch + PR
  ceremony. With it, they land work in one commit and the change
  shows up on `git log` immediately.
- **Stop condition.** Without it agents either silently overshoot the
  brief's scope OR stall without telling you. The "or when blocked"
  clause is what saves you from the quietly-burning-cycles failure
  mode.

### When to deviate from the template

- **Multi-brief ordering.** If briefs compose (e.g.,
  `kit-pump-redundancy` + `async-camera-publishers`), say "land
  [first] before starting [second]." Most current briefs are
  independent; the brief's `Out of scope` section usually flags
  relationships when they exist.
- **Mid-flight context.** If a different agent landed something on
  the branch since the brief was written, call out the recent commit
  so they sync first.
- **The brief leaves a real choice unmade.** If it lists Option A
  and Option B and you have a preference, name it in the launcher so
  they don't re-deliberate.
- **Brief's `Context bundle` is missing a module the work obviously
  needs.** Include the missing module path in the launcher. Bundle
  is meant to be exhaustive but reality drifts; close the loop in
  the brief afterward.

### Anti-patterns

- **Rewriting the brief in the launcher.** If you find yourself
  re-explaining the task in the kickoff, the brief itself is missing
  something — fix the brief, don't paper over it in the launcher.
  The brief is the durable artifact; the launcher is throwaway.
- **Overspecifying implementation.** The brief leaves implementation
  latitude on purpose; the launcher should respect it. "Use a Python
  rclpy thread" is fine guidance for a brief; "rewrite this exact
  function this way" is overspecification.
- **Forgetting the stop condition.** Even verbal kickoffs benefit
  from "and ping me when done." Without it, the agent's idea of
  "done" may be more ambitious than yours.

---

## Examples

The current queue is itself a useful set of examples:

- **Investigation** ([`real-d555-depth-range-survey.md`](active/investigations/real-d555-depth-range-survey.md))
  — open-ended diagnostic with a clear done-definition.
- **Refactor** ([`async-camera-publishers.md`](completed/async-camera-publishers.md))
  — larger surgical change with risks and out-of-scope explicit.
- **New feature** ([`jetson-headless-viewer.md`](completed/jetson-headless-viewer.md))
  — green-field work with a recommended approach + lighter alternative.
- **Bug fix** ([`completed/bridge-render-product-resolution.md`](completed/bridge-render-product-resolution.md),
  [`completed/goal-projection-depth-range.md`](completed/goal-projection-depth-range.md))
  — symptom + root-cause + acceptance criteria.
- **Convention amendment** ([`task-board-epic-structure.md`](completed/task-board-epic-structure.md))
  — meta-brief that landed at top-level `active/` rather than inside
  an epic dir; defines the layout the rest of the board uses.

Match the closest one for the kind of work you're queuing.
