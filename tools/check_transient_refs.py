#!/usr/bin/env python3
"""Fail when a tracked file names something that lives outside this repository.

The tree holds code and user-facing documentation; briefs, the board and
measurement records live in other repositories (docs/context/tasks-and-evidence.md).
Checked on every line of every tracked text file:

  path    the former in-tree homes and files of briefs and records
  date    a dated directory name (the -YYYY-MM-DD suffix records carry)
  brief   a brief's name       (only with --tasks <task-repository clone>)
  record  a record's name      (only with --records <evidence-repository clone>)
  pr      a pull-request number (listed by --list, never a failure)

Exempt: this file, and any Markdown file whose first lines carry an INTERIM
banner (a frozen document; docs/context/conventions.md). A clone given with
--tasks or --records that holds no brief or no dated directory is an error.

Without --list: prints each enforced hit and exits 1 if there is one.
With --list: prints every hit, then files / lines per file class and kind.
"""
import argparse
import collections
import pathlib
import re
import subprocess
import sys

ALLOW = {"tools/check_transient_refs.py"}
INTERIM = re.compile(r"^>\s*\*\*INTERIM")

FIXED = {
    "path": re.compile(r"docs/tasks|docs/measurements|BOARD\.md|DEFERRED_WORK"),
    "date": re.compile(r"-20[0-9]{2}-[0-9]{2}-[0-9]{2}"),
    "pr": re.compile(r"(?<![\w&#])#[0-9]{2,3}(?![\w])|pull/[0-9]+"),
}
LISTED_ONLY = {"pr"}
CLASSES = ("source", "tests", "Makefile", "tools", "guides", "context", "Readme", "other", "exempt")
DATED = re.compile(r"-20[0-9]{2}-[0-9]{2}-[0-9]{2}$")


def file_class(path):
    parts = path.split("/")
    name = parts[-1]
    if parts[0] == "source":
        test = {"tests", "test"} & set(parts[:-1]) or name.startswith("test_") or name == "conftest.py"
        return "tests" if test else "source"
    if path == "Makefile":
        return "Makefile"
    if parts[0] == "tools":
        return "tools"
    if path == "Readme.md":
        return "Readme"
    if parts[:2] == ["docs", "context"] or parts[:3] == ["docs", "tasks", "context"]:
        return "context"
    if parts[0] == "docs" and name.endswith(".md"):
        return "guides"
    return "other"


def names(pattern_words):
    words = sorted(set(pattern_words), key=len, reverse=True)
    if not words:
        return None
    return re.compile(r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(map(re.escape, words)))


def brief_names(clone):
    root = pathlib.Path(clone)
    return [p.stem for d in ("active", "parked", "completed") for p in (root / d).rglob("*.md")]


def record_names(clone):
    dirs = [p.name for p in pathlib.Path(clone).iterdir() if p.is_dir() and DATED.search(p.name)]
    return dirs + [DATED.sub("", d) for d in dirs]


def tracked(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print every hit and the counts by file class")
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent.parent))
    ap.add_argument("--tasks", help="a clone of the task repository: also check brief names")
    ap.add_argument("--records", help="a clone of the evidence repository: also check record names")
    ap.add_argument("--exclude", action="append", default=[], help="path prefix to leave out (repeatable)")
    args = ap.parse_args()

    kinds = dict(FIXED)
    if args.tasks:
        found = brief_names(args.tasks)
        if not found:
            ap.error(f"--tasks {args.tasks}: no brief under active/, parked/ or completed/")
        kinds["brief"] = names(found)
    if args.records:
        found = record_names(args.records) if pathlib.Path(args.records).is_dir() else []
        if not found:
            ap.error(f"--records {args.records}: no dated record directory")
        kinds["record"] = names(found)
    kinds = {k: rx for k, rx in kinds.items() if rx is not None}

    counts = collections.defaultdict(lambda: [set(), 0])
    failures = 0
    for path in tracked(args.root):
        if any(path.startswith(prefix) for prefix in args.exclude):
            continue
        try:
            text = (pathlib.Path(args.root) / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        frozen = path.endswith(".md") and any(INTERIM.match(line) for line in lines[:5])
        cls = "exempt" if frozen or path in ALLOW else file_class(path)
        for number, line in enumerate(lines, 1):
            for kind, rx in kinds.items():
                if not rx.search(line):
                    continue
                enforced = kind not in LISTED_ONLY and cls != "exempt"
                failures += enforced
                counts[cls, kind][0].add(path)
                counts[cls, kind][1] += 1
                if args.list or enforced:
                    print(f"{cls}\t{kind}\t{path}:{number}: {line.strip()[:160]}")

    if args.list:
        order = [k for k in ("path", "date", "brief", "record", "pr") if k in kinds]
        print("\nfiles / lines\t" + "\t".join(order))
        for cls in CLASSES:
            cells = ["%d / %d" % (len(counts[cls, k][0]), counts[cls, k][1]) for k in order]
            print(cls + "\t" + "\t".join(cells))
        return 0
    if failures:
        print(f"\n{failures} line(s) name something that lives outside this repository; "
              "see docs/context/tasks-and-evidence.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
