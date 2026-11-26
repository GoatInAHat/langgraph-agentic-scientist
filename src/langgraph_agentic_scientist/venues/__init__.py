"""Catalog of paper venues: brief, author block, template, markers and page budget.

A venue bundles everything the paper stages need to know about one paper format: the writer brief,
the canonical author block, the LaTeX template directory, the markers that identify a compiled
manuscript, the enforced page budget, and the venue-specific checks it opts into (upstream
``configs/venue/<name>.yaml`` and ``src/orchestrator/paper_venues.py``). Adding a venue is adding
a YAML file plus the referenced brief and author files; a caller can do so without touching the
package by loading a directory of its own (:meth:`VenueCatalog.from_directory`) and passing the
catalog to :class:`~langgraph_agentic_scientist.paper.validation.WorkspacePaperToolchain`.

The venue files live in this package as data; ``brief`` and ``author_block`` name files relative
to the directory that holds the YAML file (upstream named repository-relative paths).
"""

from __future__ import annotations

import importlib
import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

yaml: Any = importlib.import_module("yaml")  # PyYAML; the dev environment has no stubs for it

VENUE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

KNOWN_CHECKS = frozenset({"alife_framing"})
"""Behavioural checks implemented by the validator; a venue opts in by name."""

_SPEC_KEYS = frozenset(
    {"name", "brief", "author_block", "template_dir", "latex_markers", "page_budget", "checks"}
)
_PAGE_BUDGET_KEYS = frozenset(
    {
        "min_body_pages",
        "max_body_pages",
        "min_last_body_page_fill_ratio",
        "body_columns",
        "exclude_references",
    }
)


class VenueSpecError(ValueError):
    """A venue file is missing, malformed, or references missing assets."""


@dataclass(frozen=True)
class PageBudget:
    """Page budget enforced by the page validation gate."""

    max_body_pages: int
    min_body_pages: int | None = None
    min_last_body_page_fill_ratio: float | None = None
    body_columns: int = 1
    exclude_references: bool = False

    def as_limits(self) -> dict[str, int | float | bool]:
        """Return the mapping shape the page validation consumes."""
        limits: dict[str, int | float | bool] = {
            "max_body_pages": self.max_body_pages,
            "body_columns": self.body_columns,
            "exclude_references": self.exclude_references,
        }
        if self.min_body_pages is not None:
            limits["min_body_pages"] = self.min_body_pages
        if self.min_last_body_page_fill_ratio is not None:
            limits["min_last_body_page_fill_ratio"] = self.min_last_body_page_fill_ratio
        return limits


@dataclass(frozen=True)
class VenueSpec:
    """One paper venue as declared in ``<name>.yaml``.

    Attributes:
        name: The venue name, which is also the paper format name.
        brief: File name of the writer brief, relative to the directory of the YAML file.
        author_block: File name of the canonical author block, or ``None`` (anonymous).
        template_dir: Template directory relative to the staged ``write-paper`` skill
            (``.agents/skills/write-paper``).
        latex_markers: Regular expressions, any of which on ``paper.tex`` selects the venue.
        page_budget: The enforced page budget, or ``None`` for an advisory venue.
        checks: Behavioural checks the venue opts into.
        base: The directory that holds the YAML file, the brief and the author block.
    """

    name: str
    brief: str
    author_block: str | None
    template_dir: str
    latex_markers: tuple[str, ...]
    page_budget: PageBudget | None
    checks: frozenset[str]
    base: Traversable = field(compare=False, repr=False)

    def matches_tex(self, tex_text: str) -> bool:
        """Report whether ``tex_text`` carries one of this venue's LaTeX markers."""
        return any(re.search(marker, tex_text, re.IGNORECASE) for marker in self.latex_markers)

    def brief_text(self) -> str:
        """Return the text of the venue brief."""
        return self.base.joinpath(self.brief).read_text(encoding="utf-8")

    def author_block_text(self) -> str | None:
        """Return the text of the canonical author block, or ``None``."""
        if self.author_block is None:
            return None
        return self.base.joinpath(self.author_block).read_text(encoding="utf-8")


def _require(condition: bool, name: str, message: str) -> None:
    if not condition:
        raise VenueSpecError(f"venue {name!r}: {message}")


def _parse_page_budget(name: str, raw: Any) -> PageBudget | None:
    if raw is None:
        return None
    _require(isinstance(raw, dict), name, "page_budget must be a mapping or null")
    unknown = sorted(set(raw) - _PAGE_BUDGET_KEYS)
    _require(not unknown, name, f"page_budget has unknown keys {unknown}")
    _require("max_body_pages" in raw, name, "page_budget.max_body_pages is required")
    for key in ("max_body_pages", "min_body_pages", "body_columns"):
        value = raw.get(key)
        if value is not None:
            _require(
                type(value) is int and value > 0, name, f"page_budget.{key} must be a positive int"
            )
    ratio = raw.get("min_last_body_page_fill_ratio")
    if ratio is not None:
        _require(
            isinstance(ratio, int | float) and not isinstance(ratio, bool) and 0 < ratio <= 1,
            name,
            "page_budget.min_last_body_page_fill_ratio must lie in (0, 1]",
        )
    exclude = raw.get("exclude_references", False)
    _require(isinstance(exclude, bool), name, "page_budget.exclude_references must be a bool")
    minimum = raw.get("min_body_pages")
    _require(
        minimum is None or minimum <= raw["max_body_pages"],
        name,
        "page_budget.min_body_pages must not exceed max_body_pages",
    )
    return PageBudget(
        max_body_pages=int(raw["max_body_pages"]),
        min_body_pages=minimum,
        min_last_body_page_fill_ratio=float(ratio) if ratio is not None else None,
        body_columns=int(raw.get("body_columns", 1)),
        exclude_references=exclude,
    )


def _parse_relative_file(name: str, key: str, raw: Any, base: Traversable) -> str:
    _require(isinstance(raw, str) and bool(raw.strip()), name, f"{key} must be a non-empty string")
    parts = Path(raw).parts
    _require(
        not Path(raw).is_absolute() and ".." not in parts,
        name,
        f"{key} must be a relative path without '..'",
    )
    _require(base.joinpath(raw).is_file(), name, f"{key} {raw!r} is not a file")
    return Path(raw).as_posix()


def parse_venue_spec(name: str, raw: Any, base: Traversable) -> VenueSpec:
    """Validate a decoded venue mapping and return its :class:`VenueSpec`."""
    _require(isinstance(raw, dict), name, "venue file must contain a mapping")
    unknown = sorted(set(raw) - _SPEC_KEYS)
    _require(not unknown, name, f"unknown keys {unknown}")
    missing = sorted(_SPEC_KEYS - set(raw))
    _require(not missing, name, f"missing keys {missing}")
    _require(raw["name"] == name, name, f"name {raw['name']!r} must match the file stem")

    brief = _parse_relative_file(name, "brief", raw["brief"], base)
    author_block = (
        _parse_relative_file(name, "author_block", raw["author_block"], base)
        if raw["author_block"] is not None
        else None
    )
    template_dir = raw["template_dir"]
    _require(
        isinstance(template_dir, str)
        and bool(template_dir.strip())
        and not Path(template_dir).is_absolute()
        and ".." not in Path(template_dir).parts,
        name,
        "template_dir must be a non-empty relative path without '..'",
    )

    markers = raw["latex_markers"]
    _require(
        isinstance(markers, list)
        and bool(markers)
        and all(isinstance(m, str) and bool(m) for m in markers),
        name,
        "latex_markers must be a non-empty list of regex strings",
    )
    for marker in markers:
        try:
            re.compile(marker)
        except re.error as exc:
            raise VenueSpecError(f"venue {name!r}: latex_markers {marker!r}: {exc}") from exc

    checks = raw["checks"]
    _require(
        isinstance(checks, list) and all(isinstance(check, str) for check in checks),
        name,
        "checks must be a list of check names",
    )
    unknown_checks = sorted(set(checks) - KNOWN_CHECKS)
    _require(not unknown_checks, name, f"unknown checks {unknown_checks}")

    return VenueSpec(
        name=name,
        brief=brief,
        author_block=author_block,
        template_dir=template_dir.strip(),
        latex_markers=tuple(markers),
        page_budget=_parse_page_budget(name, raw["page_budget"]),
        checks=frozenset(checks),
        base=base,
    )


def _load_one(directory: Traversable, name: str) -> VenueSpec:
    if not VENUE_NAME_RE.fullmatch(name):
        raise VenueSpecError(f"venue name {name!r} must match {VENUE_NAME_RE.pattern}")
    path = directory.joinpath(f"{name}.yaml")
    if not path.is_file():
        raise VenueSpecError(f"venue {name!r}: {path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VenueSpecError(f"venue {name!r}: {path} is not valid YAML: {exc}") from exc
    return parse_venue_spec(name, raw, directory)


def _yaml_names(directory: Traversable) -> Iterator[str]:
    for entry in directory.iterdir():
        if entry.is_file() and entry.name.endswith(".yaml"):
            stem = entry.name[: -len(".yaml")]
            if VENUE_NAME_RE.fullmatch(stem):
                yield stem


class VenueCatalog:
    """The declared venues, keyed by name.

    ``VenueCatalog.default()`` holds the venues of the package; ``from_directory`` loads a
    directory of a caller's own, and ``with_venues`` extends a catalog.
    """

    def __init__(self, venues: Mapping[str, VenueSpec]) -> None:
        self._venues = dict(sorted(venues.items()))

    @classmethod
    def default(cls) -> VenueCatalog:
        """Return the venues shipped with the package (``iclr`` and ``alife``)."""
        return cls.from_traversable(resources.files(__package__))

    @classmethod
    def from_directory(cls, directory: Path | str) -> VenueCatalog:
        """Load every ``<name>.yaml`` of a host directory."""
        return cls.from_traversable(Path(directory))

    @classmethod
    def from_traversable(cls, directory: Traversable) -> VenueCatalog:
        """Load every ``<name>.yaml`` of a directory (a path or package resource)."""
        return cls({name: _load_one(directory, name) for name in sorted(_yaml_names(directory))})

    def with_venues(self, other: VenueCatalog) -> VenueCatalog:
        """Return a catalog holding these venues and ``other``'s (``other`` wins on a clash)."""
        return VenueCatalog({**self._venues, **other._venues})

    def names(self) -> tuple[str, ...]:
        """Return the declared venue names, which are the valid paper formats."""
        return tuple(self._venues)

    def get(self, name: str) -> VenueSpec:
        """Return venue ``name``.

        Raises:
            VenueSpecError: When the venue is not declared.
        """
        if not VENUE_NAME_RE.fullmatch(name):
            raise VenueSpecError(f"venue name {name!r} must match {VENUE_NAME_RE.pattern}")
        try:
            return self._venues[name]
        except KeyError:
            raise VenueSpecError(
                f"venue {name!r} does not exist; declared venues: {', '.join(self._venues)}"
            ) from None

    def find(self, name: str | None) -> VenueSpec | None:
        """Return the declared venue called ``name``, or ``None``."""
        return self._venues.get(name) if name is not None else None

    def has_check(self, name: str | None, check: str) -> bool:
        """Report whether venue ``name`` opts into behavioural ``check``."""
        if check not in KNOWN_CHECKS:
            raise ValueError(f"unknown venue check {check!r}")
        venue = self.find(name)
        return venue is not None and check in venue.checks

    def detect(self, tex_text: str) -> VenueSpec | None:
        """Return the first declared venue whose LaTeX markers match ``tex_text``."""
        for venue in self._venues.values():
            if venue.matches_tex(tex_text):
                return venue
        return None

    def page_limits(self) -> dict[str, dict[str, int | float | bool]]:
        """Return enforced page budgets keyed by venue name (advisory venues omitted)."""
        return {
            name: venue.page_budget.as_limits()
            for name, venue in self._venues.items()
            if venue.page_budget is not None
        }

    def page_budget_sentence(self, name: str | None) -> str:
        """Render the brief's ``{{PAPER_PAGE_BUDGET}}`` text from the enforced budget."""
        return page_budget_sentence(self.find(name))


def _sentence_templates() -> dict[str, str]:
    raw = resources.files(__package__).joinpath("page_budget.toml").read_text(encoding="utf-8")
    return {key: str(value) for key, value in tomllib.loads(raw).items()}


def page_budget_sentence(venue: VenueSpec | None) -> str:
    """Render the ``{{PAPER_PAGE_BUDGET}}`` text of a venue (upstream ``page_budget_sentence``)."""
    templates = _sentence_templates()
    budget = venue.page_budget if venue is not None else None
    if budget is None:
        return templates["no_budget"]
    if budget.min_body_pages is None:
        text = templates["up_to"].format(max=budget.max_body_pages)
    elif budget.min_body_pages == budget.max_body_pages:
        text = templates["exactly"].format(max=budget.max_body_pages)
    else:
        text = templates["between"].format(min=budget.min_body_pages, max=budget.max_body_pages)
    if budget.exclude_references:
        text += templates["exclude_references"]
    text += templates["end"]
    if budget.min_body_pages is not None and budget.min_last_body_page_fill_ratio is not None:
        text += templates["fill_rule"].format(
            min=budget.min_body_pages, ratio=f"{budget.min_last_body_page_fill_ratio:.0%}"
        )
    return text


def default_catalog() -> VenueCatalog:
    """Return :meth:`VenueCatalog.default`."""
    return VenueCatalog.default()


__all__ = [
    "KNOWN_CHECKS",
    "VENUE_NAME_RE",
    "PageBudget",
    "VenueCatalog",
    "VenueSpec",
    "VenueSpecError",
    "default_catalog",
    "page_budget_sentence",
    "parse_venue_spec",
]
