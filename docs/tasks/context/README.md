# `context/` — reusable agent-onboarding modules

This directory holds **stable facts about the system as it is now**,
factored out of the per-task briefs in `docs/tasks/`. The goal is to
let task briefs stay focused on the task itself rather than re-state
the same preamble every time.

If you're picking up a task, the brief's `## Context bundle` section
will list the modules to read first. Read those before starting; they
cover the things the brief assumes you already know.

If you're authoring a new task, see
[`docs/tasks/README.md`](../README.md) for how to compose context.

If you're updating a context module — typically as part of landing a
task that invalidated a fact — keep reading.

---

## What belongs in a context module

- **Stable facts about the system as it is now**, not as it was or
  will be.
- **Cross-cutting truths** that show up in multiple task briefs. If
  the same fact has only ever appeared in one brief, it's not
  context yet — leave it in the brief.
- **Pointers, not exposition.** When the answer to "where is X?" is
  a file path, link the file path. Don't re-explain the contents.
- **Agent-facing tone**: same voice as the task briefs. Terse,
  declarative, concrete, link-heavy.

## What does NOT belong in a context module

- **History** ("we used to do X but switched to Y in commit abc123").
  Git tracks that.
- **Speculation or future-state** ("we plan to refactor this into
  Z"). Goes in a task brief, not in context.
- **Currently open bugs or in-flight work**. Goes in `docs/tasks/` as
  a brief. Context describes the *intended invariants*; tasks
  describe the *current deviations from them*.
- **Operator-facing procedures**. The cheatsheet at
  [`docs/example_commands_cheatsheet.md`](../../example_commands_cheatsheet.md)
  is the home for "commands operators copy-paste."
- **Anything time-bound.** No "as of 2026-04-27" preambles.
  Modules describe present state; if the present state changes, the
  module changes.

---

## Module size

**Soft ceiling: ~150 lines.** Context modules are reading material
agents pull up before starting work; they need to be skimmable in a
minute or two. If a module is past 150 lines, one of these is true:

- It's documentation, not context. Move it (e.g., to
  `docs/SIM_TO_REAL_TUNING_GUIDE.md`-style long-form docs) and link
  to it from the slimmer module.
- It bundles two distinct topics. Split it.
- It includes exposition that should be a link to source. Replace
  the exposition with the link.

---

## How tasks reference modules

Task briefs include a small section near the top:

```markdown
## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
```

That replaces what used to be 50 lines of duplicated preamble. The
brief itself stays focused on what's specific to *this* task.

---

## Maintenance contract

The acceptance criteria of every task brief includes:

```markdown
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.
```

This is the forcing function that keeps modules current. The agent
that ships the change is the one closest to the new truth.

In practice:

1. While implementing a task, you change something — a default value,
   a script flag, a runtime invariant.
2. Before opening the commit, scan the modules referenced in your
   task's `## Context bundle` for any sentence the change makes
   wrong. If found, edit the module in the same commit.
3. The commit ends up touching both the source change AND the
   context module. That's fine; that's the point.

A secondary safety net: when the operator stages a fresh integration
round (currently tracked as
[`docs/tasks/integration-prompts-refresh.md`](../integration-prompts-refresh.md)),
that audit task should re-read context modules end-to-end and flag
anything stale.

---

## When to add a new module

Add a new context module when:

- The same fact (a paragraph or so worth of explanation) has shown
  up in 3+ task briefs OR is about to.
- Or: you're writing a brief that needs preamble that isn't covered
  by any existing module, AND a future agent on a different task
  would also need it.

Don't add a module:

- Just-in-case for content nobody is asking for yet.
- For one-off facts specific to one task.
- For history. The repo's git log handles that.

When you add a module, also add a one-line entry below in the
[Current modules](#current-modules) section so the index stays
accurate.

---

## When to split / merge / retire a module

- **Split** when a module pushes past 150 lines AND covers more
  than one topic. Always link the descendants to each other so
  cross-references survive.
- **Merge** when two modules are routinely pulled together by every
  brief that pulls either, and merging would land both under the
  ceiling.
- **Retire** when the system has changed enough that the module
  no longer describes anything real. Delete it, and audit
  remaining briefs for now-dangling links (use grep:
  `grep -rn 'context/<retired-name>.md' docs/tasks/`).

---

## Current modules

- [`repo-topology.md`](repo-topology.md) — hosts, branch, conda envs,
  workspace layout, key entry-point scripts.
- [`ownership-boundaries.md`](ownership-boundaries.md) — DGX vs
  Jetson lanes, the peer-agent rule, shared-boundary append-only
  files.
- [`bridge-runtime-invariants.md`](bridge-runtime-invariants.md) —
  bridge architecture, cmd_vel normalization contract, sim-time
  timeout, headless-vs-`--viz kit` defaults.

Two more candidates flagged for Pass 2 if duplication shows up:

- `conventions.md` — commit style, no-transient-doc-references rule,
  mermaid edge-label rule.
- `end-to-end-flow.md` — operator → planner → executor → grounding →
  goal projection → Nav2 → bridge → Isaac Sim arrow chain.

---

## Style guide for module authors

- **Lead with the fact, not the rationale.** Why something is true
  is sometimes useful, but the fact itself is what an agent needs in
  the first sentence.
- **Use links liberally.** "Defined in `path/file.py:42`" beats
  "the value lives in a file called file.py."
- **Use small tables for enumerations.** Hosts × IPs, owners × file
  paths, etc.
- **Avoid bulletpoint chains.** A 12-bullet list is hard to skim;
  a 3-row table is fast.
- **No marketing.** Don't say "the bridge is robust"; say "the bridge
  publishes /clock at 50 Hz from a Python rclpy thread."
- **Match the existing modules' tone.** Pull one up before
  authoring a new one and keep voice consistent.
