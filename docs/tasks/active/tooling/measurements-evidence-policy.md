# Keep `docs/measurements/` to the record prose

**Type:** docs / convention change (plus the retroactive sweep that applies it)
**Owner:** DGX (the records were all written on the sim host, and the deposits
are pushed from it)
**Priority:** P2 — nothing is blocked, but the directory grows by one payload
set per measurement session and every session makes the next reviewer's job
worse. The cost is paid at review time and at clone time, by everyone.
**Estimate:** S (one session: deposit, slim, write the page; no hardware, no
suite run)
**Branch:** `task/measurements-evidence-policy`

## Story

As a **reader of a measurement record**, I want **the record directory to hold
the record and nothing else**, so that **the prose that carries the finding is
what I meet when I open it, and the payloads it rests on are one clone away
with their digests attached rather than 137 files deep in the diff.**

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`docs/tasks/README.md`](../../README.md) — the brief lifecycle this ships
  under

## Context

`docs/measurements/` was given a tracked home in `20da03a` with one convention,
stated only in that commit body: one subdirectory per measurement session, the
raw JSONL, and a README saying what the numbers were. Six records later the
directory holds 147 tracked files, and the two largest records are 81 and 34
files against one README each. Three problems come with that:

- **Volume defeats review.** A payload set is not reviewable by reading it, and
  a reviewer who cannot read it cannot vouch for what is in it — including for
  content that should not be public.
- **Staleness.** Probe scripts pin the era they were written in. They stop
  running as the tree moves, and nothing tells a reader which ones still do.
- **Readability.** The directory is a public-facing surface, and the prose that
  is the point of it is buried.

The fix is already in use and undocumented: `goal-a-attribution-2026-08-22` and
the two Stage 3 records deposit their bulk evidence in the companion evidence
repository and cite it by sha256. What is missing is the rule, written down,
and the older records brought to it.

The rule this brief lands: a record directory holds `README.md`, plus
`provenance.md` where the record keeps one, and nothing else. Everything else
is deposited under the same record name in the evidence repository, in a
`record-files/` directory laid out exactly as the record directory was, and the
record's README cites the deposit by repository URL, directory, commit hash and
the sha256 of every file. The mirror is what keeps the records re-runnable: the
scripts in the two largest records resolve their inputs as
`docs/measurements/<record>/…` relative to the working directory, so one
`cp -a` restores the directory and every command the README shows runs
unchanged.

## Acceptance criteria

- [ ] `docs/measurements/README.md` states the policy: what stays, what is
      deposited, deposit-before-record ordering, immutability, the >100 MB
      rule, and how a reviewer verifies a record.
- [ ] Every record directory holds only `README.md` and, where it has one,
      `provenance.md`.
- [ ] Every deposited file is in the evidence repository under its record name,
      with a `DEPOSIT.md` carrying host, source, date and per-file sha256, and
      each deposit verifies from its own digest list.
- [ ] Each slimmed record's README carries an evidence section naming the
      repository URL, the deposit directory, the deposit commit hash and the
      sha256 of every deposited file, and the digests in the README equal the
      digests in the deposit.
- [ ] No load-bearing number in any record README or `provenance.md` changes.
- [ ] No relative link in a record points at a file that left the tree.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- **A history rewrite.** Removal governs the working tree; the removed files
  stay in git history, and rewriting it is a separate decision that is not
  taken here.
- **An enforcement lever.** Nothing stops a payload being committed into a
  slimmed record. An ignore rule or a CI size check is the mechanism, and it is
  a follow-up.
- **Re-running anything.** No measurement is repeated and no verdict is
  revisited; the records' findings are carried across unchanged.
