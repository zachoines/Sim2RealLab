# Measurement records

A measurement record is one directory here. It holds the record — what was
measured, how, every load-bearing number, the verdicts — and nothing else.

## What lives in this repository

- `README.md`, the record itself.
- `provenance.md`, where the record keeps one: hosts, trees, interpreter,
  artifact digests, and the inputs this repository does not carry.

That is the whole in-tree record. Everything else a record produced or read —
analysis and probe scripts, JSON / JSONL / CSV results, junit XML, `.npz` and
`.npy` arrays, logs, videos, images — lives in the companion evidence
repository, `https://github.com/zachoines/Sim2RealLab-Artifacts`, under a
directory of the same name as the record. That repository is private; a record
here is written to stand on its own for a reader who cannot open it, and the
digests are what let a reader who can confirm it.

The split is there because payload files and era-pinned scripts are not
reviewable at the volume a session produces, they go stale as the tree moves
around them, and they bury the prose a reader comes here for. Prose reviews;
payloads verify. Each belongs where its own check works.

## Depositing

The deposit lands first and the record PR follows, so the record can cite the
deposit by content rather than by promise.

- A record's own file set is deposited under `<record>/record-files/`, laid
  out exactly as the record directory was. A path the record's prose names is
  the same path in the deposit. Evidence that never lived in the record —
  host-side captures, logs pulled off a machine — gets its own sibling
  directory under the same record name.
- A deposit is one directory and one act of depositing, and it carries a
  `DEPOSIT.md`: what the files are, the host and source they came from, the
  date, and the sha256 of every file, taken over the uncompressed content where
  a file is stored compressed.
- Deposits are immutable once a merged record references one. A correction is a
  new sibling directory with its own `DEPOSIT.md` saying which deposit it
  supersedes and why, and the record's evidence section moves to citing it. The
  superseded deposit stays where it is, because merged records still point at
  it.
- A file over GitHub's 100 MB limit is gzip-compressed or split before deposit,
  with both the stored and the uncompressed digest recorded.

## Citing a deposit from a record

The record's README carries an evidence section naming four things for each
deposit it rests on: the repository URL, the deposit directory, the deposit
commit hash, and the sha256 of every file in it. The commit hash is what makes
the citation content-stable — it names bytes, not a location that can move
underneath the record. A record with more than one deposit — its own file set
in `record-files/`, a host-side capture beside it — cites each of them, and
says which holds what.

Prose that used to link a file in the record directory names it as a plain path
instead. The evidence section is the one place that says where those paths now
resolve, so a reader meets the answer once rather than at every mention.

## Verifying a record

Clone the evidence repository alongside this checkout and restore the record
directory from its deposit. Because the deposit mirrors the record, the
commands a README shows then run unchanged:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/<record>/record-files/. docs/measurements/<record>/
```

The deposit's own `DEPOSIT.md` comes across with the files. It is not part of
the record, and neither it nor the restored files belong in a commit — nothing
in `.gitignore` stops that, so a restored record directory wants clearing
before the next commit rather than staging.

The digests check without restoring anything, from the deposit itself:

```
cd Sim2RealLab-Artifacts/<record>/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

The same digests are listed in the record's evidence section, so a reviewer
who has both trees can confirm that the record cites the bytes the deposit
holds.
