# Branching and PRs

The repo follows a **one brief → one branch → one PR** rule. Long-lived
shared branches (like the historical `phase_15-isaaclab3`) are not used
anymore.

## Naming

Branch name is the brief's filename minus the `.md`, prefixed with
`task/`:

| Brief | Branch |
|---|---|
| [`docs/tasks/vlm-bbox-overlay.md`](../vlm-bbox-overlay.md) | `task/vlm-bbox-overlay` |
| [`docs/tasks/async-camera-publishers.md`](../async-camera-publishers.md) | `task/async-camera-publishers` |
| [`docs/tasks/branch-per-task-convention.md`](../branch-per-task-convention.md) | `task/branch-per-task-convention` |

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
8. Move the brief into `docs/tasks/completed/` in the same PR (or in a
   tiny follow-up PR if you only got the implementation hash after
   merge). Stamp the brief per [`conventions.md`'s closed-brief
   lifecycle](conventions.md#closed-task-brief-lifecycle), and add the
   PR URL on a `**PR:**` line below `**Status:**`.

## PR composition

- **One PR per brief.** Don't bundle two unrelated briefs into one PR.
- **Adjacent fixes that surface during the work** (e.g., a one-line
  config bug discovered while validating the feature) are OK to bundle
  if and only if they would block the brief from being verifiable.
  Otherwise: open a separate brief + branch + PR.
- **PR title** mirrors the brief's title. Keep it under ~70 chars.
- **PR body** structure: `## Summary` (1–3 bullets), `## Test plan`
  (markdown checklist), no agent-attribution footer.

## What does NOT use this convention

- **`docs/archive/` / `docs/tasks/completed/`** — historical record,
  never edited in flight.
- **Single-commit emergency hotfixes that need to land in <1 hour.**
  The convention is for planned work. A hotfix that bypasses brief
  authoring still goes through a PR off `main`, just without the brief.
- **Shared experimental branches with no merge target.** If a branch
  is being used to share code between agents without intent to merge,
  that's not a task; it's a side channel. Don't create those.

## Why we changed

Pre-convention (through commit `0592e21`), `phase_15-isaaclab3`
accumulated 109 commits / +31k LOC / 183 files across 3 months of
unrelated work — Isaac Lab migration, bridge work, autonomy polish,
docs system, the headless visualizer, the SLAM floor-leak fix.
Reviewing it as one PR is hard; reverting any one piece is harder.
The goal of the convention is that no future PR is that big.
