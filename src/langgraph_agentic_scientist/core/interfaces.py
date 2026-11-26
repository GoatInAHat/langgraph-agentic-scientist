"""Protocols of replaceable verification and outside services, with their record types.

The phase packages ship the default implementations (``LocalTaskGate``,
``HarnessRubricVerifier``, ``SemanticScholarRetriever``, ``WorkspacePaperToolchain``,
``AutoillustrateRenderer``); a caller substitutes any of them through
:class:`~langgraph_agentic_scientist.ScientistContext`. The protocols live in the core so that
the context can name them without importing a phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from .state import RubricSpec, RubricStatus, StateModel, TaskRef

if TYPE_CHECKING:
    from ..evaluation.protocol import EvaluationResult
    from ..workspace.protocol import Workspace
    from .results import ValidationReport

GateCheck = Literal[
    "preflight",
    "acceptance-smoke",
    "contract",
    "runtime",
    "search-smoke",
    "final-smoke",
    "production-entry",
    "qualification",
]
"""The checks of upstream's harness-owned task gate (``task.sh`` gate operations)."""


class GateReport(StateModel):
    """The report of one gate check.

    Attributes:
        check: The check that ran.
        passed: Whether the scaffold passed.
        infrastructure: True when the check failed for a host reason, which upstream retries
            without spending a repair.
        category: Upstream's typed failure category.
        stage: The stage of the gate that failed.
        diagnostic: Bounded diagnostic text for repair prompts.
        fingerprint: The scaffold fingerprint the check ran against.
        details: Further JSON data of the check.
    """

    check: str
    passed: bool
    infrastructure: bool = False
    category: str | None = None
    stage: str | None = None
    diagnostic: str = ""
    fingerprint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class TaskGate(Protocol):
    """The drafting gates: contract preflight, smoke, production entry and qualification."""

    async def run_check(
        self,
        check: GateCheck,
        *,
        task: TaskRef,
        workspace: Workspace,
        evaluation_id: str,
    ) -> GateReport:
        """Run one check of the scaffold at ``task.root`` and return its report.

        ``evaluation_id`` keys the evaluations the check runs, so that a resumed check does not
        evaluate twice.
        """
        ...


@runtime_checkable
class RubricVerifier(Protocol):
    """Writes the rubric status of a harness-evaluated candidate (upstream ``_rubric_status``)."""

    async def verify(
        self,
        *,
        rubrics: Sequence[RubricSpec],
        evaluation: EvaluationResult,
        program_dir: str,
        workspace: Workspace,
        attempt_id: str,
        idea_id: str,
    ) -> RubricStatus:
        """Return the rubric status of the evaluated program."""
        ...


class ReferenceQuery(StateModel):
    """One literature query of a reference plan."""

    query: str
    purpose: str = ""
    limit: int = Field(10, ge=1)
    filters: dict[str, Any] = Field(default_factory=dict)


class ReferenceRecord(StateModel):
    """One retrieved reference with its provenance."""

    key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    bibtex: str | None = None
    source: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ReferenceRetriever(Protocol):
    """Finds literature for a query (upstream: Semantic Scholar, then OpenAlex and scholarly)."""

    async def search(self, query: ReferenceQuery, *, workspace: Workspace) -> list[ReferenceRecord]:
        """Return references for one query, best first."""
        ...


class CompileResult(StateModel):
    """The outcome of compiling a paper."""

    success: bool
    pdf_path: str | None = None
    log: str = ""
    page_count: int | None = None


@runtime_checkable
class PaperToolchain(Protocol):
    """Validation gates, LaTeX compilation and PDF text extraction of the paper phase."""

    async def validate(
        self, paper_dir: str, *, workspace: Workspace, options: dict[str, Any]
    ) -> ValidationReport:
        """Run the validation gates of ``validate_and_compile_paper`` on a paper directory."""
        ...

    async def compile(self, paper_dir: str, *, workspace: Workspace) -> CompileResult:
        """Compile ``paper.tex`` into a PDF."""
        ...

    async def pdf_text(self, pdf_path: str, *, workspace: Workspace) -> str:
        """Extract the text of a PDF."""
        ...


class FigureAvailability(StateModel):
    """Which concept-figure backends are available (fills ``{{PAPER_*_API_KEY_CONTEXT}}``)."""

    openai: bool = False
    gemini: bool = False
    openai_context: str = "No concept-figure renderer is configured for this run."
    gemini_context: str = "No concept-figure renderer is configured for this run."


class ConceptFigureRequest(StateModel):
    """A request to render the concept figure of a paper."""

    paper_dir: str
    instructions: str
    mode: Literal["full", "medium", "light"] = "full"
    output_path: str = "figures/concept.png"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptFigureResult(StateModel):
    """The outcome of rendering a concept figure."""

    success: bool
    path: str | None = None
    detail: str = ""


@runtime_checkable
class ConceptFigureRenderer(Protocol):
    """Renders concept figures with an outside tool (upstream: Autoillustrate)."""

    def availability(self) -> FigureAvailability:
        """Return which backends are available, for the writer prompts."""
        ...

    async def render(
        self, request: ConceptFigureRequest, *, workspace: Workspace
    ) -> ConceptFigureResult:
        """Render a concept figure into the paper directory."""
        ...


__all__ = [
    "CompileResult",
    "ConceptFigureRenderer",
    "ConceptFigureRequest",
    "ConceptFigureResult",
    "FigureAvailability",
    "GateCheck",
    "GateReport",
    "PaperToolchain",
    "ReferenceQuery",
    "ReferenceRecord",
    "ReferenceRetriever",
    "RubricVerifier",
    "TaskGate",
]
