# `docs/tasks/` — work queue + authoring guide

This directory holds **one-shot, agent-pickup-able work briefs** in a
Jira-style format. Each `.md` here describes a single discrete piece of
work that a fresh agent (DGX, Jetson, or either) can pick up and
execute against without needing the operator to also write a long
preamble each time.

Briefs are persistent across sessions. When work lands, the brief is
typically deleted (or marked done in its frontmatter) in the same
commit that ships the change. **Briefs are not historical records** —
git history serves that purpose. Once a task is shipped, its brief
has done its job.

If you've landed here as a fresh agent looking for work, scan the
list and pick a task whose `Owner` matches your host.

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

- `kit-pump-redundancy-investigation.md`
- `async-camera-publishers.md`
- `jetson-headless-viewer.md`
- `integration-prompts-refresh.md`

Avoid prefixes like `001-`, dates, or owner initials. The `Owner` /
`Priority` / `Estimate` fields in the frontmatter convey ordering and
ownership; the filename should still read sensibly when you scan
`ls docs/tasks/`.

---

## Brief format

```markdown
# <One-line title — verb-leading where it makes sense>

**Type:** task / bug / investigation / refactor / docs
**Owner:** DGX agent / Jetson agent / either
**Priority:** P0 / P1 / P2
**Estimate:** S / M / L  (rough effort; ~hours / ~day / ~multi-day)

## Story

As a **<role>**, I want **<action>**, so that **<outcome>**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
  *(omit modules that are not relevant to this task)*

## Context

<background, code pointers, hypotheses, and the symptom that
triggered this task — anything specific to THIS task that the
context modules don't already cover. Keep stable facts in modules;
keep this section about the task itself.>

## Acceptance criteria

- [ ] <criterion 1 — concrete, falsifiable>
- [ ] <criterion 2>
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.
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

- **Required:** Title, the four frontmatter fields, Story, Mission /
  Context (one or the other), Acceptance criteria, Out of scope.
- **Recommended:** Context bundle (skip only for trivial,
  self-contained tasks).
- **Optional:** Investigation pointers (skip when there's nothing
  to point at). Open questions, risks, ranked-hypotheses tables —
  add when the task warrants.

### The acceptance bullet that keeps context current

Every brief must include this bullet (or an equivalent):

```markdown
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.
```

This is the forcing function that prevents context drift. The agent
that ships the change is the one closest to the new truth — easier
to update one fact in one module than to ripple it through half a
dozen briefs later.

---

## How to compose context for a new task

1. **Identify which modules cover preamble you'd otherwise duplicate.**
   Skim [`context/`](context/). If your brief would otherwise repeat
   "DGX is at 192.168.50.196 / branch is phase_15-isaaclab3 / conda
   env is env_isaaclab3" — that's a `context/repo-topology.md` link.

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

### Anti-patterns

- **Status briefs.** "Investigation of why X is slow" with no
  Acceptance criteria. If there's no done-definition, it's not a
  task — it's a note. Put it in `docs/PERF_INVESTIGATION_*.md` or
  similar.
- **Multi-task briefs.** "Refactor A and also fix B and also
  document C" — split into three briefs. The Jira-style format is
  designed for one ticket = one PR.
- **Stale briefs.** When work ships, delete or close the brief in
  the same commit. Living briefs that nobody owns become noise.

---

## Examples

The current queue is itself a useful set of examples:

- **Investigation** ([`kit-pump-redundancy-investigation.md`](kit-pump-redundancy-investigation.md))
  — open-ended diagnostic with a clear done-definition.
- **Refactor** ([`async-camera-publishers.md`](async-camera-publishers.md))
  — larger surgical change with risks and out-of-scope explicit.
- **New feature** ([`jetson-headless-viewer.md`](jetson-headless-viewer.md))
  — green-field work with a recommended approach + lighter alternative.
- **Docs** ([`integration-prompts-refresh.md`](integration-prompts-refresh.md))
  — refresh of existing artifacts, with concrete drift listed.
- **Bug fix** ([`bridge-render-product-resolution.md`](bridge-render-product-resolution.md),
  [`goal-projection-depth-range.md`](goal-projection-depth-range.md))
  — symptom + root-cause + acceptance criteria.

Match the closest one for the kind of work you're queuing.
