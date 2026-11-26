#!/usr/bin/env python3
"""Fail when a repository file names a local path or an internal reference.

The repository describes only itself and its upstream. This check scans every file that git
tracks or would track (untracked files not ignored by ``.gitignore``), except the vendored upstream
files under ``third_party/``, for host paths (home and temporary directories), working-directory
names of development tools, and citation marks of internal notes. CI runs it on every push.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ("third_party/",)
# Assembled from parts so that this file does not match its own patterns.
PATTERNS = {
    "a macOS home path": re.compile("/" + "Users" + "/"),
    "a temporary host path": re.compile("/" + "private" + "/"),
    "a Linux home path": re.compile("/" + "home" + "/"),
    "a development tool's working directory": re.compile("scratch" + "pad", re.IGNORECASE),
    "an internal citation mark": re.compile(r"\[" + "G" + r"\d\d"),
}


def candidate_files() -> list[str]:
    """Return the files git tracks or would track, relative to the repository root."""
    output = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted({line for line in output.splitlines() if line})


def scan(paths: list[str]) -> list[str]:
    """Return one problem line per match."""
    problems: list[str] = []
    for relative in paths:
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    problems.append(f"{relative}:{number}: {label}")
    return problems


def main() -> int:
    """Run the check."""
    problems = scan(candidate_files())
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"{len(problems)} problem(s): the repository must not name local paths", file=sys.stderr
        )
        return 1
    print("public check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
