"""The prompt registry: upstream prompts, skills and workflow texts as package data.

Every prompt file of the package is derived from an upstream file by the changes declared in
``prompts/changes/*.toml`` (see ``tools/derive_prompts.py``); the files under ``prompts/harness/``
are the package's own texts. :class:`PromptRegistry` loads them, lets a caller override any entry,
and renders a template the way upstream does:

1. the six shared-recipe markers (``{{EVALUATOR_EVIDENCE_CONTRACT}}``,
   ``{{CANDIDATE_ENVIRONMENT_RECIPE}}``, ``{{BASELINE_FAILURE_RECIPE}}``,
   ``{{EVALUATOR_FEASIBILITY_RECIPE}}``, ``{{WORKER_DIAGNOSTICS_RECIPE}}``,
   ``{{REPAIR_DIAGNOSTICS_POLICY}}``) are replaced by the fragment texts, in that order, as
   upstream ``prompt_contract.expand_evidence_contract`` does (skipped for the design prompts, as
   ``render_design_prompt`` skips them);
2. ``{{KEY}}`` placeholders are substituted in one pass over the template: a substituted value is
   never scanned again (upstream's sequential ``str.replace`` re-expanded placeholders inside
   values); unknown placeholders stay literal, as upstream leaves them.

Keys are package-relative paths without the ``.md`` suffix (``"fragments/evaluator_evidence"``,
``"investigation/harness_edit"``); files with another suffix keep it
(``"workflow/investigation/context_blocks.toml"``). Skill files are keyed
``"skills/<name>/<path>"`` (for example ``"skills/memory-log/SKILL.md"``).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType

from ..core.errors import PromptNotFound

FRAGMENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("{{EVALUATOR_EVIDENCE_CONTRACT}}", "fragments/evaluator_evidence"),
    ("{{CANDIDATE_ENVIRONMENT_RECIPE}}", "fragments/candidate_environment"),
    ("{{BASELINE_FAILURE_RECIPE}}", "fragments/baseline_failures"),
    ("{{EVALUATOR_FEASIBILITY_RECIPE}}", "fragments/evaluator_feasibility"),
    ("{{WORKER_DIAGNOSTICS_RECIPE}}", "fragments/worker_diagnostics"),
    ("{{REPAIR_DIAGNOSTICS_POLICY}}", "fragments/repair_diagnostics"),
)
"""Upstream's fragment markers and their registry keys, in upstream's expansion order."""

PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")
"""A ``{{KEY}}`` placeholder."""

_EXCLUDED_PROMPT_DIRS = ("changes",)


def _iter_files(root: Traversable, prefix: str = "") -> Iterator[tuple[str, Traversable]]:
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        name = child.name
        if name.startswith(".") and name != ".agents":
            continue
        if name == "__pycache__":
            continue
        relative = f"{prefix}{name}"
        if child.is_dir():
            yield from _iter_files(child, f"{relative}/")
        elif child.is_file():
            yield relative, child


def _prompt_key(relative: str) -> str:
    return relative[:-3] if relative.endswith(".md") else relative


def package_prompt_files() -> dict[str, bytes]:
    """Return every prompt file shipped with the package, keyed as the registry keys them."""
    base = resources.files(__package__ or "langgraph_agentic_scientist.prompts")
    files: dict[str, bytes] = {}
    for relative, item in _iter_files(base):
        top = relative.split("/", 1)[0]
        if top in _EXCLUDED_PROMPT_DIRS or relative == "__init__.py" or relative.endswith(".py"):
            continue
        files[_prompt_key(relative)] = item.read_bytes()
    return files


def package_skill_files() -> dict[str, bytes]:
    """Return every file of every skill shipped with the package, keyed ``skills/<name>/<path>``."""
    base = resources.files("langgraph_agentic_scientist").joinpath("skills")
    if not base.is_dir():
        return {}
    return {f"skills/{relative}": item.read_bytes() for relative, item in _iter_files(base)}


def render_text(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``{{KEY}}`` placeholders in one pass; unknown placeholders stay literal."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = values.get(key)
        return match.group(0) if value is None else str(value)

    return PLACEHOLDER_RE.sub(replace, template)


def placeholders(template: str) -> set[str]:
    """Return the placeholder names that occur in a template."""
    return set(PLACEHOLDER_RE.findall(template))


class PromptRegistry:
    """Prompt, skill and workflow texts, overridable entry by entry.

    Use :meth:`default` for the package's texts and :meth:`with_overrides` to replace entries.
    A registry is immutable; overriding returns a new registry. Overrides are part of the prompt
    fingerprint that a run records (:meth:`fingerprint`), so a resumed run must be given the same
    overrides.
    """

    def __init__(self, texts: Mapping[str, str], skill_files: Mapping[str, bytes]) -> None:
        self._texts = MappingProxyType(dict(texts))
        self._skill_files = MappingProxyType(dict(skill_files))

    @classmethod
    def default(cls) -> PromptRegistry:
        """Return the registry of the package's own texts (``importlib.resources`` data)."""
        texts = {key: data.decode("utf-8") for key, data in package_prompt_files().items()}
        skills = package_skill_files()
        for key, data in skills.items():
            if key.endswith(".md"):
                texts[key] = data.decode("utf-8")
        return cls(texts, skills)

    def with_overrides(self, overrides: Mapping[str, str | os.PathLike[str]]) -> PromptRegistry:
        """Return a registry in which the given keys are replaced.

        A value is the replacement text, or a path (``os.PathLike``) whose content is read as
        UTF-8. Keys must exist; a typo raises :class:`PromptNotFound` instead of being ignored.
        """
        texts = dict(self._texts)
        skill_files = dict(self._skill_files)
        for key, value in overrides.items():
            if key not in texts:
                raise PromptNotFound(key)
            text = (
                Path(value).read_text(encoding="utf-8") if isinstance(value, os.PathLike) else value
            )
            texts[key] = text
            if key in skill_files:
                skill_files[key] = text.encode("utf-8")
        return PromptRegistry(texts, skill_files)

    def keys(self) -> list[str]:
        """Return every key, sorted."""
        return sorted(self._texts)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys, sorted."""
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._texts

    def text(self, key: str) -> str:
        """Return the raw text of an entry."""
        try:
            return self._texts[key]
        except KeyError as exc:
            raise PromptNotFound(key) from exc

    def expand_fragments(self, template: str) -> str:
        """Replace the six fragment markers by their texts, in upstream's order."""
        for marker, key in FRAGMENT_MARKERS:
            if marker in template:
                template = template.replace(marker, self.text(key))
        return template

    def render(self, key: str, values: Mapping[str, str], *, fragments: bool = True) -> str:
        """Render an entry: expand fragments (unless ``fragments=False``), then fill values."""
        template = self.text(key)
        if fragments:
            template = self.expand_fragments(template)
        return render_text(template, values)

    def skill_names(self) -> list[str]:
        """Return the names of the skills shipped with the registry."""
        return sorted({key.split("/", 2)[1] for key in self._skill_files})

    def skill_files(self, name: str) -> dict[str, bytes]:
        """Return the files of skill ``name`` keyed by their path inside the skill directory."""
        prefix = f"skills/{name}/"
        files = {
            key[len(prefix) :]: data
            for key, data in self._skill_files.items()
            if key.startswith(prefix)
        }
        if not files:
            raise PromptNotFound(f"skills/{name}")
        return files

    def digests(self, keys: Iterable[str] | None = None) -> dict[str, str]:
        """Return the SHA-256 of each text and skill file (all when ``keys`` is ``None``)."""
        selected = set(self._texts) | set(self._skill_files) if keys is None else set(keys)
        result: dict[str, str] = {}
        for key in sorted(selected):
            if key in self._skill_files:
                data = self._skill_files[key]
            else:
                data = self.text(key).encode("utf-8")
            result[key] = hashlib.sha256(data).hexdigest()
        return result

    def fingerprint(self) -> str:
        """Return one SHA-256 over every text and skill file, after overrides."""
        hasher = hashlib.sha256()
        for key, digest in self.digests().items():
            hasher.update(key.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        return hasher.hexdigest()


__all__ = [
    "FRAGMENT_MARKERS",
    "PLACEHOLDER_RE",
    "PromptRegistry",
    "package_prompt_files",
    "package_skill_files",
    "placeholders",
    "render_text",
]
