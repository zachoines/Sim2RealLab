# Branching and PRs

The repo follows a **one brief → one branch → one PR** rule. Each
piece of planned work lives on its own short-lived task branch off
`main`; long-lived shared feature branches are not used.

## Naming

Branch name is the brief's filename (basename, minus the `.md`),
prefixed with `task/`. The epic subdirectory does **not** leak into
the branch name:

| Brief (in the task repository) | Branch |
|---|---|
| `active/<epic>/<brief-slug>.md` | `task/<brief-slug>` |
| `parked/<epic>/<brief-slug>.md`, once picked up | `task/<brief-slug>` |

The brief's frontmatter carries the predetermined name on a `**Branch:**`
line so a fresh agent doesn't have to guess. The brief's own housekeeping
rides a branch of the same name in the task repository.

## Branch-off point

Always `main`. Don't branch off another in-flight task branch — if
your work depends on Task A, wait for A's PR to merge and rebase or
re-branch off main. Branching off branches creates merge tangles and
defeats the point of small, independently-mergable PRs.

## Workflow per task

1. Pick a brief whose `**Owner:**` matches your host.
2. `git checkout main && git pull`.
3. `git checkout -b task/<brief-slug>` matching the brief's `**Branch:**`.
4. Work the brief. Commit per the rules in
   [`conventions.md`](conventions.md#commit-messages) (scoped subjects,
   no agent-attribution trailers).
5. Push: `git push -u origin task/<brief-slug>`.
6. Open the PR with `gh pr create --base main --head task/<brief-slug> ...`.
   `gh` is installed on both DGX and Jetson; `gh auth login` once per host.
7. After review, **merge with a merge commit** (not squash, not rebase).
   The granular history per change is what makes blame, bisect, and
   revert work the way you'd expect.
8. Ship the brief in the task repository, **in a pull request opened
   there before this PR merges** — not as a follow-up afterward. It
   stamps the brief (using this branch's work commit as `<ship-commit>`),
   moves it to `completed/<YYYY>/` and updates its board rows; each of
   the two bodies names the other. The summary is [`conventions.md`'s
   closed-brief lifecycle](conventions.md#closed-task-brief-lifecycle);
   the full sequence is the task repository README's "Shipping a brief:
   order of operations".

## PR composition

- **One PR per brief.** Don't bundle two unrelated briefs into one PR.
- **Adjacent fixes that surface during the work** (e.g., a one-line
  config bug discovered while validating the feature) are OK to bundle
  if and only if they would block the brief from being verifiable.
  Otherwise: open a separate brief + branch + PR.

### PR register — the title and body are the public merge record

A PR outlives the workflow that produced it. Write the title and body
for a maintainer with repo access and no knowledge of how the work was
coordinated. The process layer — who asked for what, what a work
session did or lost, how a decision was adjudicated — lives in the
brief's dated narrative, never on the PR.

**Title.** States what merging changes, imperative mood, ≤ ~70 chars,
with the load-bearing mechanism or consequence when it fits. When the
brief's title already does this, mirror it; when the brief title names
an activity, the PR title names the tree change instead. Models from
this repo:

- "Drive inference off depth arrival to hold the 30 Hz cadence"
- "Take the depth block median so far-clip straddling can't invent depths"
- "Collapse the deploy config levels to one key, one home"

Not titles: process verbs (escalate, adjudicate, re-validate, hand
back), verdicts with no change named ("X stands permanently"),
epic/arm tokens, dates.

**Body.** Five sections, all present (a one-line section is fine):

- `## Summary` — the defect or gap, the change, the mechanism. Present
  tense, third person, self-contained.
- `## Evidence` — suites with counts and gates run; a measurement's
  result in one paragraph, with the rule it was read by. A checklist is
  welcome (`## Test plan` remains an accepted name for code PRs).
  Measurement dates are fine; work sessions are not characters in the
  story. Digests and other repositories' commits stay in References.
- `## Scope` — what is deliberately untouched, defaults changed,
  rollback lever, known residuals.
- `## Docs` — the guides and context modules the change touched
  (`docs/guides/`, `docs/context/`, `Readme.md`). One line each.
- `## References` — the only cross-repository links, one line per item
  in a fixed shape; a line kind that does not apply is left out:

  ```
  Tasks: <brief-name> — Sim2RealLab-Tasks PR #<n>
  Record: Sim2RealLab-Artifacts@<commit> <record-dir>/
  ```

  `Tasks:` names each brief the PR ships or files, with the
  task-repository pull request that carries its stamp, move and board
  row. `Record:` names each measurement record the PR rests on or
  produced, at the evidence-repository commit that holds it. Nothing
  else in the body links across repositories
  ([`tasks-and-evidence.md`](tasks-and-evidence.md)).

**Register rules** (title, body, commits, and review comments alike):

- Third person throughout. No second-person address, offers, or
  questions to a reader ("say the word", "happy to add", "if you
  want") — decisions are made before the PR opens; genuinely open
  questions go to the review thread, phrased against the code.
- No internal role or workflow vocabulary: coordinator, operator,
  dispatch, ruling, handback, session report, pre-registration
  scoring, arm/cell labels.
- No paths outside the repo (home directories, scratchpads,
  machine-local reports). Evidence a PR relies on is a record in the
  evidence repository, cited in References, or is reproduced in the
  body. Where files live on a machine, the record's provenance says
  so; the PR cites the record.
- No tribunal register: "the ruling is", "the question is closed",
  "measured not assumed", "attack lines that failed". State the fact
  and its evidence; confidence is carried by the evidence, not by
  oaths.
- No attribution footers on the body ("Generated with …",
  `Co-Authored-By:`) — the same rule
  [commits follow](conventions.md#commit-messages).
- The body is the merge record: corrections found in review are edited
  into the body, not appended as dialogue, and at merge the body must
  agree with every artifact the PR ships (a body never asserts what
  its own diff retracts).

The skeleton lives at
[`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md).

## What does NOT use this convention

- **Briefs, the board and measurement records.** They live in other
  repositories and follow those repositories' rules
  ([`tasks-and-evidence.md`](tasks-and-evidence.md)).
- **Single-commit emergency hotfixes that need to land in <1 hour.**
  The convention is for planned work. A hotfix that bypasses brief
  authoring still goes through a PR off `main`, just without the brief.
- **Shared experimental branches with no merge target.** If a branch
  is being used to share code between agents without intent to merge,
  that's not a task; it's a side channel. Don't create those.

## Why this convention

Long-lived shared branches accumulate unrelated work — migrations,
bridge changes, autonomy polish, docs sweeps — and become impossible
to review or revert as a unit. The one-brief-per-PR rule keeps every
change small enough to review in isolation, blame cleanly, and
revert without dragging in adjacent work.
