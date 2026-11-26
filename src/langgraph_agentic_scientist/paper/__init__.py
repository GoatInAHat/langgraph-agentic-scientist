"""The paper phase: data, references, validation and the writing loop.

:func:`build_paper_graph` returns the phase as a compiled graph. It runs with the results of the
earlier phases, with a prebuilt :class:`~langgraph_agentic_scientist.core.results.PaperData`, or in
``edit`` and ``revision`` modes on an existing paper. Its replaceable slots are ``paper.data``,
``paper.references``, ``paper.postdraft_enrichment``, ``paper.figures``, ``paper.validation``,
``paper.audit`` and ``paper.review_gate``.
"""

from ..core.interfaces import ConceptFigureRenderer, PaperToolchain, ReferenceRetriever
from ..core.results import EditOptions, PaperData, PaperResult
from .audit import AuditInput, AuditOutput, build_audit_graph
from .figures import AutoillustrateRenderer
from .graph import build_paper_graph
from .models import (
    AuditFinding,
    AuditPayload,
    AuditResult,
    InLoopReview,
    PaperInput,
    PaperMode,
    PaperOutput,
    PaperState,
    PaperStrategy,
    PosthocRanking,
    ReviewPayload,
    StrategyAnswer,
    WriterFamily,
)
from .references import SemanticScholarRetriever, build_references_graph
from .review_gate import ReviewGateInput, ReviewGateOutput, build_review_gate_graph
from .validation import WorkspacePaperToolchain
from .writer import PostdraftInput, PostdraftOutput, build_postdraft_graph

__all__ = [
    "AuditFinding",
    "AuditInput",
    "AuditOutput",
    "AuditPayload",
    "AuditResult",
    "AutoillustrateRenderer",
    "ConceptFigureRenderer",
    "EditOptions",
    "InLoopReview",
    "PaperData",
    "PaperInput",
    "PaperMode",
    "PaperOutput",
    "PaperResult",
    "PaperState",
    "PaperStrategy",
    "PaperToolchain",
    "PostdraftInput",
    "PostdraftOutput",
    "PosthocRanking",
    "ReferenceRetriever",
    "ReviewGateInput",
    "ReviewGateOutput",
    "ReviewPayload",
    "SemanticScholarRetriever",
    "StrategyAnswer",
    "WorkspacePaperToolchain",
    "WriterFamily",
    "build_audit_graph",
    "build_paper_graph",
    "build_postdraft_graph",
    "build_references_graph",
    "build_review_gate_graph",
]
