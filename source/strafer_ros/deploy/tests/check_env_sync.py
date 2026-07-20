#!/usr/bin/env python3
"""Sync-pin: the compose env_file mirrors must not drift from canon.

The compose lane uses plain KEY=VALUE env_file mirrors because docker's
`env_file` cannot source the canonical bash env files (which use `$(...)` for a
self-locating CYCLONEDDS_URI). This test compares OVERLAPPING keys and fails on
drift, so the mirrors can't silently diverge from strafer_bringup/config/env_*.env.

Pairs (mirror <- canon):
  deploy/compose/sim.env       <- strafer_bringup/config/env_sim_in_the_loop.env
  deploy/compose/autonomy.env  <- strafer_bringup/config/env_autonomy.env

Run:  python3 deploy/tests/check_env_sync.py     (exit 0 = in sync)
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # deploy/tests
DEPLOY = HERE.parent                                         # deploy
CANON = DEPLOY.parent / "strafer_bringup" / "config"         # strafer_ros/strafer_bringup/config

PAIRS = [
    (DEPLOY / "compose" / "sim.env", CANON / "env_sim_in_the_loop.env"),
    (DEPLOY / "compose" / "autonomy.env", CANON / "env_autonomy.env"),
]


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v


def parse(path: Path, canonical: bool) -> dict:
    pat = re.compile(r"^export\s+([A-Z0-9_]+)=(.*)$" if canonical else r"^([A-Z0-9_]+)=(.*)$")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pat.match(line)
        if m:
            out[m.group(1)] = _unquote(m.group(2))
    return out


def main() -> int:
    failures, checked = [], 0
    for mirror, canon in PAIRS:
        mv = parse(mirror, canonical=False)
        cv = parse(canon, canonical=True)
        for key in sorted(set(mv) & set(cv)):
            c = cv[key]
            # canonical values using shell expansion (e.g. self-locating
            # CYCLONEDDS_URI) can't be compared literally -- skip them.
            if "$(" in c or "${" in c:
                continue
            checked += 1
            if mv[key] != c:
                failures.append(f"{mirror.name}: {key}={mv[key]!r} != canon({canon.name})={c!r}")
    if failures:
        print("ENV SYNC DRIFT (update the mirror or canon to match):")
        for f in failures:
            print("  " + f)
        return 1
    print(f"env mirrors in sync with canon ({checked} overlapping keys checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
