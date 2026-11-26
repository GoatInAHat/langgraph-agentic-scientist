#!/usr/bin/env python3
"""Copy the vendored upstream files from a local AI-Scientist-v3 checkout and verify them.

The upstream files this repository carries are listed in
``third_party/ai-scientist-v3/MANIFEST.toml``: a ``[selection]`` table of git pathspecs, and
one ``[[file]]`` entry per selected file with its SHA-256 and, for files that live outside
``third_party/`` (the examples), its destination.

Usage::

    python tools/vendor_upstream.py --upstream PATH_TO_CHECKOUT            # copy and verify
    python tools/vendor_upstream.py --upstream PATH_TO_CHECKOUT --refresh  # rebuild file list
    python tools/vendor_upstream.py --check                                # verify the copies

Files are read with ``git show <commit>:<path>`` from the commit pinned in
``third_party/ai-scientist-v3/UPSTREAM_COMMIT``, so uncommitted changes in the checkout never
leak into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
MANIFEST = THIRD_PARTY / "MANIFEST.toml"
COMMIT_FILE = THIRD_PARTY / "UPSTREAM_COMMIT"
UPSTREAM_URL = "https://github.com/SakanaAI/AI-Scientist-v3"

# Files copied outside third_party/: upstream prefix -> repository prefix.
DESTINATION_PREFIXES = {
    "examples/circpack/": "examples/circpack/",
    "examples/circpack_task_drafting/": "examples/circpack_task_drafting/",
}


@dataclass(frozen=True)
class ManifestFile:
    """One vendored file: its upstream path, digest and repository destination."""

    path: str
    sha256: str
    dest: str


def pinned_commit() -> str:
    """Return the upstream commit the repository is pinned to."""
    return COMMIT_FILE.read_text(encoding="utf-8").strip()


def destination_for(path: str) -> str:
    """Return the repository-relative destination of an upstream path."""
    for prefix, target in DESTINATION_PREFIXES.items():
        if path.startswith(prefix):
            return target + path[len(prefix) :]
    return f"third_party/ai-scientist-v3/{path}"


def load_manifest() -> tuple[list[str], list[ManifestFile]]:
    """Read the selection pathspecs and the file entries of the manifest."""
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = [str(item) for item in data.get("selection", {}).get("paths", [])]
    files = [
        ManifestFile(
            path=str(entry["path"]),
            sha256=str(entry["sha256"]),
            dest=str(entry.get("dest") or destination_for(str(entry["path"]))),
        )
        for entry in data.get("file", [])
    ]
    return selection, files


def _git(upstream: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(upstream), *args], check=True, capture_output=True
    ).stdout


def list_selected(upstream: Path, commit: str, selection: list[str]) -> list[str]:
    """Expand the selection pathspecs to the files they name at the pinned commit."""
    output = _git(upstream, "ls-tree", "-r", "--name-only", commit, "--", *selection)
    return sorted(line for line in output.decode("utf-8").splitlines() if line)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_manifest(selection: list[str], files: list[ManifestFile], commit: str) -> None:
    """Write the manifest in a stable, reviewable layout."""
    lines = [
        "# Upstream files vendored unchanged from AI-Scientist-v3 (Apache License 2.0, Sakana AI).",
        "# Regenerate with: python tools/vendor_upstream.py --upstream <checkout> --refresh",
        f"upstream = {_toml_string(UPSTREAM_URL)}",
        f"commit = {_toml_string(commit)}",
        "",
        "[selection]",
        "paths = [",
        *(f"    {_toml_string(item)}," for item in selection),
        "]",
        "",
    ]
    for entry in files:
        lines.append("[[file]]")
        lines.append(f"path = {_toml_string(entry.path)}")
        lines.append(f"sha256 = {_toml_string(entry.sha256)}")
        if entry.dest != f"third_party/ai-scientist-v3/{entry.path}":
            lines.append(f"dest = {_toml_string(entry.dest)}")
        lines.append("")
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def _safe_dest(dest: str) -> Path:
    parts = PurePosixPath(dest).parts
    if not parts or PurePosixPath(dest).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe manifest destination: {dest}")
    return REPO_ROOT.joinpath(*parts)


def copy_files(upstream: Path, commit: str, files: list[ManifestFile]) -> int:
    """Copy every manifest file from the checkout and verify its digest."""
    errors = 0
    for entry in files:
        data = _git(upstream, "show", f"{commit}:{entry.path}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            print(f"digest mismatch for {entry.path}: {digest} != {entry.sha256}", file=sys.stderr)
            errors += 1
            continue
        target = _safe_dest(entry.dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return errors


def check_files(files: list[ManifestFile]) -> int:
    """Verify that every vendored copy in the repository matches its manifest digest."""
    errors = 0
    for entry in files:
        target = _safe_dest(entry.dest)
        if not target.is_file():
            print(f"missing vendored file: {entry.dest}", file=sys.stderr)
            errors += 1
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != entry.sha256:
            print(f"vendored file differs from upstream: {entry.dest}", file=sys.stderr)
            errors += 1
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, help="local AI-Scientist-v3 git checkout")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild the file list and digests from the selection pathspecs",
    )
    parser.add_argument("--check", action="store_true", help="only verify the vendored copies")
    args = parser.parse_args(argv)

    commit = pinned_commit()
    selection, files = load_manifest()
    if args.check:
        errors = check_files(files)
        print(f"{len(files)} vendored files checked, {errors} problem(s)")
        return 1 if errors else 0
    if args.upstream is None:
        parser.error("--upstream is required unless --check is given")
    upstream = args.upstream.expanduser().resolve()
    if args.refresh:
        paths = list_selected(upstream, commit, selection)
        files = [
            ManifestFile(
                path=path,
                sha256=hashlib.sha256(_git(upstream, "show", f"{commit}:{path}")).hexdigest(),
                dest=destination_for(path),
            )
            for path in paths
        ]
        write_manifest(selection, files, commit)
    errors = copy_files(upstream, commit, files)
    print(f"{len(files)} files vendored from {commit[:12]}, {errors} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
