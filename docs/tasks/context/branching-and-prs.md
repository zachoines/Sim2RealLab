# Branching and PRs

The repo follows a **one brief → one branch → one PR** rule. Each
piece of planned work lives on its own short-lived task branch off
`main`; long-lived shared feature branches are not used.

## Naming

Branch name is the brief's filename (basename, minus the `.md`),
prefixed with `task/`. The epic subdirectory does **not** leak into
the branch name:

| Brief | Branch |
|---|---|
| [`docs/tasks/active/harness/mission-generator.md`](../active/harness/mission-generator.md) | `task/mission-generator` |
| [`docs/tasks/active/multi-room/autonomy-stack.md`](../active/multi-room/autonomy-stack.md) | `task/autonomy-stack` |
| [`docs/tasks/completed/task-board-epic-structure.md`](../completed/task-board-epic-structure.md) | `task/task-board-epic-structure` |
| [`docs/tasks/completed/branch-per-task-convention.md`](../completed/branch-per-task-convention.md) | `task/branch-per-task-convention` |

The brief's frontmatter carries the predetermined name on a `**Branch:**`
line so a fresh agent doesn't have to guess.

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
8. Move the brief into `docs/tasks/completed/` **inside this PR, before
   merge** — not as a follow-up afterward. Stamp it (using the branch's
   work commit as `<ship-commit>`) per [`conventions.md`'s closed-brief
   lifecycle](conventions.md#closed-task-brief-lifecycle); the full
   sequence is [`docs/tasks/README.md`'s "Shipping a brief: order of
   operations"](../README.md#shipping-a-brief-order-of-operations).

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
an activity, the PR title names the tree change instead (a findings PR
*records*; a brief-only PR *files*). Models from this repo:

- "Drive inference off depth arrival to hold the 30 Hz cadence"
- "Take the depth block median so far-clip straddling can't invent depths"
- "Collapse the deploy config levels to one key, one home"

Not titles: process verbs (escalate, adjudicate, re-validate, hand
back), verdicts with no change named ("X stands permanently"),
epic/arm tokens, dates.

**Body.** Four sections, all present (a one-line section is fine):

- `## Summary` — the defect or gap, the change, the mechanism. Present
  tense, third person, self-contained.
- `## Evidence` — suites with counts, measurements with the setup that
  produced them, gates run. A checklist is welcome (`## Test plan`
  remains an accepted name for code PRs). Measurement dates are fine;
  work sessions are not characters in the story.
- `## Scope` — what is deliberately untouched, defaults changed,
  rollback lever, known residuals.
- `## Docs` — the brief lifecycle line (stamped + moved + BOARD) and
  any follow-up briefs filed, by name. One line each; no board
  choreography.

**Register rules** (title, body, commits, and review comments alike):

- Third person throughout. No second-person address, offers, or
  questions to a reader ("say the word", "happy to add", "if you
  want") — decisions are made before the PR opens; genuinely open
  questions go to the review thread, phrased against the code.
- No internal role or workflow vocabulary: coordinator, operator,
  dispatch, ruling, handback, session report, pre-registration
  scoring, arm/cell labels.
- No paths outside the repo (home directories, scratchpads,
  machine-local reports). Evidence a PR relies on is in the repo — in
  the brief or findings doc it links — or reproduced in the body.
  Where artifacts live on a machine, the brief records it; the PR
  cites the brief.
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
[`.github/PULL_REQUEST_TEMPLATE.md`](../../../.github/PULL_REQUEST_TEMPLATE.md).

## What does NOT use this convention

- **`docs/archive/` / `docs/tasks/completed/`** — historical record,
  never edited in flight.
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
