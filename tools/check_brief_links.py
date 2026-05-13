#!/usr/bin/env python3
"""Walk every Markdown file under docs/tasks/ and report broken local links.

A link is "broken" if its target is a relative path that doesn't resolve to
an existing file (or directory) on disk. External URLs (http/https/mailto)
and pure anchor fragments are ignored. Anchors after `#` are stripped before
resolution.

Exit 0 on a clean sweep, 1 if any broken link is found. Intended for CI /
pre-commit / local sanity checks after the post-reorg link sweep.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)]+)\)")
# Inline backtick spans (`...`) — links inside them are prose examples, not
# real links, so they're stripped before link extraction.
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "ftp://")


def is_local(target: str) -> bool:
    target = target.strip()
    if not target:
        return False
    if target.startswith("#"):
        return False
    if target.startswith(EXTERNAL_PREFIXES):
        return False
    return True


def split_anchor(target: str) -> str:
    return target.split("#", 1)[0]


def iter_links(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    in_fence = False
    fence_marker = ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        # Strip inline backtick spans so quoted examples don't false-flag.
        scrubbed = INLINE_CODE_RE.sub(lambda _m: " " * len(_m.group(0)), line)
        for m in LINK_RE.finditer(scrubbed):
            yield lineno, m.group("text"), m.group("target").strip()


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    scan_root = repo_root / "docs" / "tasks"
    if not scan_root.exists():
        print(f"scan root not found: {scan_root}", file=sys.stderr)
        return 2

    broken: list[tuple[Path, int, str, str, Path]] = []
    files = sorted(scan_root.rglob("*.md"))
    for md in files:
        for lineno, _text, target in iter_links(md):
            if not is_local(target):
                continue
            path_part = unquote(split_anchor(target))
            if not path_part:
                continue
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                broken.append((md, lineno, target, path_part, resolved))

    if not broken:
        print(f"OK — {len(files)} files scanned, no broken links under {scan_root.relative_to(repo_root)}/")
        return 0

    print(f"BROKEN — {len(broken)} broken link(s) across {len({b[0] for b in broken})} file(s):")
    print()
    for md, lineno, target, path_part, resolved in broken:
        rel = md.relative_to(repo_root)
        print(f"  {rel}:{lineno}")
        print(f"    target: {target}")
        print(f"    resolves to (missing): {resolved.relative_to(repo_root) if resolved.is_relative_to(repo_root) else resolved}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
