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
`docs/tasks/completed/`** in the same commit (or in an immediate
follow-up commit so the Shipped stamp can carry the implementation
commit's SHA). Stamp the top of the moved brief with:

```
**Status:** Shipped <YYYY-MM-DD> in `<commit-sha>` (<host>).
**Follow-ups:** [`<follow-up.md>`](../<follow-up.md>) — short hook.
```

`<host>` is `DGX`, `Jetson`, or `Either`. `<follow-up.md>` is
optional. See `docs/tasks/completed/goal-projection-depth-range.md`
for the canonical example.

Update `docs/tasks/README.md`'s Examples section if it linked the
brief at the top level — the link should now point at
`completed/<brief>.md`.

Do **not** delete the brief outright. Git history records what
changed; the brief records what we set out to do, the acceptance
criteria we held the change to, and which follow-ups it spawned —
those are different artifacts and both are useful.
