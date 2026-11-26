# Tasks and evidence live outside this repository

This repository holds code and user-facing documentation. Briefs and the board live in `Sim2RealLab-Tasks`;
records and their evidence in `Sim2RealLab-Artifacts` (large files in `Sim2RealLab-Artifacts-bulk`). Seven rules.

## 1. The tree names nothing transient

No file here — code, test, config, guide or context module — links to or names a brief, the
board, a measurement record, a deposit or a dated measurement. A comment
states the present fact and its mechanism. A context module that would need to name briefs is an
epic map and belongs in the task repository. The general form of
[`conventions.md`'s rule](conventions.md#no-transient-documentation-references-in-code).
`make docs-check` enforces the part a pattern can see, in every tracked file but the checker itself and
documents under an INTERIM banner: always the former in-tree homes of briefs and records, the board's
and the deferred-work list's file names and dated directory names; brief and record names only when
clones of the two other repositories are given (`make docs-check TASKS_CLONE=<path> ARTIFACTS_CLONE=<path>`).

## 2. The PR body is the only cross-reference

The register keeps `## Summary`, `## Evidence`, `## Scope`, `## Docs`
([`branching-and-prs.md`](branching-and-prs.md)) and ends with a fixed fifth block:

```
## References
Tasks: <brief-name> — zachoines/Sim2RealLab-Tasks#<n>   one line per brief shipped or filed
Record: Sim2RealLab-Artifacts@<commit> <record-dir>/    one line per record the PR rests on or produced
```

Nothing else in the body links across repositories. The Evidence section quotes a record's result
in one paragraph, with the rule it was read by, and the suite counts; digests and commits stay in
the References line.

## 3. Where a change lands

A measurement that changes no code, config, default or guide is a record in the evidence
repository and a board row in the task repository; it opens no PR here. A code change a
measurement motivated cites the record in References. A correction to a published record is a
dated line in that record and, if it changes a decision, a task-repository PR on the brief or the
board.

## 4. The evidence repository is written directly

Agents push to its `main`: linear history, the repository's checker green, the commit subject
naming the record directory first. A record is immutable once a PR here cites it or its index
lists it; corrections are dated lines and sibling deposits. The evidence repository's README is
the policy.

## 5. The task repository is reviewed

Agents open PRs there (filing, shipping, parking, board rows); the maintainer merges. A board row is
one line; narrative goes to the brief or the record. A pure measurement's board-row PR carries the
record's Result block, and that is its review.

## 6. Documentation maintenance

The same-commit sweep ([`conventions.md`](conventions.md#user-facing-documentation-maintenance))
covers `docs/guides/`, `docs/context/` and `Readme.md`. Briefs and records are maintained in their
own repositories by their own rules.

## 7. Pre-registration

A measurement's plan is committed to the evidence repository before its first run; amendments are
dated and committed before the runs they cover; the record's findings are numbered against the
registered tests.
