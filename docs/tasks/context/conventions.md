# Conventions

Cross-cutting rules every agent on this repo must follow. Each item
here exists because the rule was breached at least once and the fix
was annoying.

---

## Commit messages

Plain commit messages. **No `Co-Authored-By:` trailer, no
`🤖 Generated with ...` line, no agent attribution of any kind.**
The author field is the operator; the commit body should read like a
human wrote it.

The commit body **may** reference:

- Task brief filenames in `docs/tasks/` (`bridge-render-product-resolution.md`).
- Sibling commit hashes for context (`fixes a regression from b6e46e9`).
- Issue or PR identifiers if external trackers exist.

The commit body **must not** reference:

- The model that produced the change.
- The session or chat that produced it.
- "Generated with" / "Authored by Claude" / similar.

Style: scoped subject (e.g., `fix(strafer_lab/bridge): ...`,
`docs(tasks): ...`). Subject ≤ ~70 chars. Imperative mood. Body
explains *why* and lists the surgical pieces of the change. Match
the tone of `git log --oneline -20` for whichever subtree the change
lives in.

---

## Comment style

**Default to no comments.** Only write one when the WHY is non-obvious
and wouldn't be recovered by a careful read of the surrounding code.
When a comment is warranted, one short sentence (rarely two) is the
target — a hidden constraint, a subtle invariant, a workaround for a
specific bug, or behavior that would surprise a reader.

**Comments must not contain:**

- History (`replaced the old DistanceController`, `previously gated on
  envelope_factor`, `this used to use stock Nav2 BT`).
- Brief / task / PR / commit references — covered by
  [`#no-transient-documentation-references-in-code`](#no-transient-documentation-references-in-code).
- Broad architectural explanations — those belong in a context
  module under `docs/tasks/context/` or the relevant design doc.
- Restatement of what the code does. Identifier names already do that.
- Speculation about future work (`could later be extended to ...`).

**Why:** comments rot. Architecture prose that names absent files,
knobs, or briefs goes stale silently and misleads the next reader.
Short comments scoped to local non-obvious facts age gracefully
because they're tied to the code immediately below them.

**How to apply:**

- When in doubt, delete the comment. The diff should be smaller than
  what you'd reflexively write.
- If you find yourself describing the design, stop. Move the prose
  to a context module (or omit it — the design likely speaks for
  itself once the comment is gone).
- The XML / YAML / docstring at the top of a config file follows the
  same rule as inline code comments. A file header isn't a free pass
  to paste architecture.
- Counter-example pattern to keep handy: instead of "this BT
  differs from Nav2's stock in two places: <history>", write a
  single line on the one BT.CPP gotcha a future tuner would
  re-introduce if not warned. Same content density, fraction of
  the prose.

This is the project-wide rule; section
[`#no-transient-documentation-references-in-code`](#no-transient-documentation-references-in-code)
below is the narrower, older sibling that called out one specific
kind of rot.

---

## No transient documentation references in code

**Docstrings, comments, CLI `--help` text, error messages, log
strings, and test titles must not reference transient project
artifacts.** Banned phrasing:

- `Task N`, `DGX Task N`, `Jetson Task N`, `post-Task-N`.
- `Phase 15`, `phase_15`, `PHASE_15_*.md`, install validation
  phases (`Phase A1`, `Phase B4`, ...).
- `Phase 1` / `Phase 2` used to label workstream splits.
- `Section 5.5.4`, `STRAFER_AUTONOMY_NEXT.md`, design-doc section
  numbers.
- Specific commit hashes inside source.
- Branch names inside source (`phase_15`, `main`).

**Why:** task names rename, briefs get deleted, branches merge,
section numbers shift. Source files must read clean a year from now;
the docs that named the work-in-flight at the time the code was
written will not.

**How to apply:**

- When deferring functionality, describe *what* is deferred and
  *why a placeholder is in place* — not "phase 2 will do this."
  Example: instead of `"Phase 2 wires the runtime adapter"`, write
  `"The runtime adapter that wires this to a live Isaac Sim env is
  added separately; this module ships the pure-Python orchestrator
  only."`
- When pointing at a sibling module, link by **module path**
  (`strafer_lab.bridge.graph`), not by task or phase.
- For schemas / output formats produced by other scripts, name
  **the script** (`extract_scene_metadata.py`), not the task that
  created it.
- Commit messages and task briefs in `docs/tasks/` **may**
  reference task IDs and commit hashes — those live in git and in
  the brief queue, not in source files.

Sweep before every commit:

```
rg -n "Task \d+|Phase [A-D]?\d|phase_15|Section \d+\.\d+|post-Task" path/to/changed/files
```

---

## Mermaid edge labels

Mermaid's parser treats `(`, `)`, `[`, `]`, `{`, `}` as node-shape
wrappers even inside edge-label pipes `A -->|text| B`. Any of those
characters in an edge label causes a render failure with a cryptic
`got 'PS'` / `got 'PE'` / `got 'SQS'` / `got 'SQE'` error that names
the token class, not the offending character.

**Substitutes:**

- Rephrase with em-dash: `— deferred: X` instead of `(deferred: X)`.
- Drop delimiters: `run_export(...)` becomes `run_export ...`.
- For numeric ranges: `0..1000` instead of `[0, 1000]`.
- If the literal character is truly required, escape with HTML
  entities: `&#40;` `&#41;` `&#91;` `&#93;` `&#123;` `&#125;`.

After writing or editing a Mermaid diagram, sweep the source for
shape-delimiter characters inside pipes:

```
rg '\|[^|]*[()[\]{}][^|]*\|' path/to/file.md
```

Applies to any Mermaid in this repo (design docs, runbooks, task
briefs, READMEs).

---

## Closed task brief lifecycle

When a brief in `docs/tasks/` ships, **move it into
`docs/tasks/completed/` and stamp it inside the shipping PR, before
merge** — never as a follow-up afterward. The exact sequence is
[`docs/tasks/README.md`'s "Shipping a brief: order of operations"](../README.md#shipping-a-brief-order-of-operations);
this section is the cross-cutting summary, that section is canonical.

Stamp the top of the moved brief with:

```
**Status:** Shipped <YYYY-MM-DD> in `<ship-commit>` (<host>).
**PR:** <github-pr-url>
**Follow-ups:** [`<follow-up.md>`](../active/<epic>/<follow-up.md>) — short hook.
```

`<ship-commit>` is the **work commit on the branch** (the one that
lands the substantive change), not the merge commit — the merge
commit doesn't exist at PR-creation time, and stamping it was the
recurring "backfill the SHA after merge" step that kept getting
forgotten. `<host>` is `DGX`, `Jetson`, or `Either`. `<follow-up.md>`
is optional. The `**PR:**` line is mandatory for work that landed via
the [task → branch → PR convention](branching-and-prs.md); omit it
only for briefs whose work pre-dates that convention. Historical
entries that carry a merge SHA, or none, are fine — don't churn them
for format consistency.

Update `docs/tasks/README.md`'s Examples section if it linked the
brief at the top level — the link should now point at
`completed/<brief>.md`.

Do **not** delete the brief outright. Git history records what
changed; the brief records what we set out to do, the acceptance
criteria we held the change to, and which follow-ups it spawned —
those are different artifacts and both are useful.

---

## Task board structure

Active and parked briefs live under epic subdirs. `completed/` and
`context/` stay flat. Full layout, naming rules, and the un-park
workflow are documented in
[`docs/tasks/README.md`'s `## Directory layout`](../README.md#directory-layout).

**The fixed epic set.** The nine epics in use today are
`multi-room`, `trained-policy`, `harness`, `clip-validation`,
`sim-performance`, `reliability`, `tooling`, `experimental`,
`investigations`. This is a **fixed small set**, not free-form
tagging. Adding a tenth epic — or splitting / merging an existing
one — is itself a convention change: file a brief that amends the
layout in [`README.md`](../README.md), gets reviewed like any
other PR, and ships with the rename commits inside it. Do **not**
invent a `misc/` bucket; the urge to do so is the signal that the
brief might belong somewhere else, or that a new epic is genuinely
needed.

**Un-parking is a `git mv`.** A brief in `parked/<epic>/` becomes
pickable by `git mv parked/<epic>/<brief>.md
active/<epic>/<brief>.md` in the PR that picks it up, plus a
[`BOARD.md`](../BOARD.md) row update in the same commit. No
frontmatter status field, no separate ceremony — the path is the
state.

**Branch naming is unchanged.** Per
[`branching-and-prs.md`](branching-and-prs.md), branches are
`task/<basename>`. The epic subdir does **not** leak into the
branch name. `active/harness/teleop-driver.md` → branch
`task/teleop-driver`.

---

## User-facing documentation maintenance

The brief acceptance template already includes:

```
If your work invalidates a fact in any referenced context module,
update that module in the same commit.
```

That covers `docs/tasks/context/`. It does **not** cover the rest of
the repo's user-facing surfaces, and those drift fastest when a brief
ships behavior the surfaces previously described as missing,
deferred, or shaped differently. The contract here extends the
maintenance rule beyond context modules.

**Before opening (or merging) the PR for a brief, sweep these surfaces
and update any that your change invalidates — in the same PR.**

| Surface | What to check |
|---|---|
| Top-level [`Readme.md`](../../../Readme.md) | "What ships today", "Repository structure" (`Scripts/` listing), "Run", "Deferred / known limitations". |
| [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md) | "Scripts and tools inventory", "Contracts", "Testing", "Deferred / known limitations". |
| [`source/strafer_ros/README.md`](../../../source/strafer_ros/README.md) | Package list, launch / config surface, topic / action contracts. |
| [`source/strafer_autonomy/README.md`](../../../source/strafer_autonomy/README.md) | Skill table, planner endpoints, executor commands. |
| [`source/strafer_vlm/README.md`](../../../source/strafer_vlm/README.md) | Endpoint table, env-var / config surface. |
| [`docs/example_commands_cheatsheet.md`](../../example_commands_cheatsheet.md) | Operator one-liners — copy-paste fidelity matters. |
| [`docs/INTEGRATION_*.md`](../..) | DGX / Jetson / sim-in-the-loop spin-up runbooks. |
| [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../SIM_TO_REAL_TUNING_GUIDE.md), [`docs/SYSTEM_FLOW_DIAGRAMS.md`](../../SYSTEM_FLOW_DIAGRAMS.md) | Cited-by-name reference docs. |
| [`docs/tasks/DEFERRED_WORK.md`](../DEFERRED_WORK.md) | Items you just shipped must be **deleted from this file** in the same commit, per its own maintenance contract at the bottom of the file. |

### Trigger heuristics — your change probably invalidates user-facing docs if it:

- Adds, renames, or removes a `Scripts/` or `source/<pkg>/scripts/`
  entry point (the top-level `Readme.md` and the owning package
  README list these by name).
- Changes a CLI surface that operators copy-paste — flags, defaults,
  required args, mutual exclusion (the cheatsheet and runbooks
  embed these).
- Adds, renames, or removes a public function / class / contract
  named in a "What ships today" or "Contracts" listing.
- Ships something previously listed under "Deferred / known
  limitations" or in `DEFERRED_WORK.md` — those entries must move
  or be deleted.
- Changes test layout, testing entry points, or how to run the
  suite (the `Testing` section in package READMEs commits to a
  specific path / runner).
- Changes a topic name, action name, env var, or config field that
  any README documents.

### Sweep before commit

Pick keywords specific to your change and grep all candidate surfaces
in one shot — empty match = nothing to update:

```
rg -nE "<your-script-name>|<your-flag>|<your-function>|<your-env-var>" \
   Readme.md docs/ source/*/README.md
```

If a hit exists, update the surface in the same commit / PR. The
maintenance contract is identical to the context-module rule: a doc
that lies about the system is worse than no doc, and the agent
shipping the change is closest to the new truth.

### Acceptance-bullet upgrade

New briefs should use the broader template line:

```markdown
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
```

Existing briefs authored under the narrower text are not invalidated
by this change — they pick up the broader contract through this
context module on the next ship sweep.
