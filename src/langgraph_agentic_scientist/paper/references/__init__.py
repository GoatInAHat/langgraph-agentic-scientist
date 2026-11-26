"""References: plan the literature search, retrieve references and write the bibliography.

``build_references_graph`` is the paper phase's step 2 (see
:mod:`~langgraph_agentic_scientist.paper.references.graph`). The retrieval service is replaceable
(:class:`~langgraph_agentic_scientist.core.interfaces.ReferenceRetriever`); the default is
:class:`SemanticScholarRetriever`, which also offers upstream's OpenAlex and scholarly fallbacks.
"""

from .bibtex import extract_bibtex_key, load_bibtex_rows
from .diagnostics import archive_round, record_reference_round, reference_repair_context
from .gaps import (
    validate_draft_reference_gaps_payload,
    write_draft_reference_gaps_validation_feedback,
)
from .graph import (
    ReferencesInput,
    ReferencesOutput,
    ReferencesResult,
    build_references_graph,
)
from .merge import merge_reference_artifacts, merge_reference_payloads
from .plan import (
    MAX_REFERENCE_QUERY_COUNT,
    VALID_QUERY_CATEGORIES,
    ReferencePlan,
    ReferencePlanAnswer,
    ReferenceQueryPlan,
    reference_query_plan_metadata,
    reference_query_texts,
    validate_reference_query_plan_payload,
    write_plan_files,
    write_reference_query_plan_validation_feedback,
)
from .retrieve import (
    FallbackOutcome,
    reference_provenance_payload,
    retrieval_failure_message,
    run_script_fallbacks,
    write_reference_outputs,
)
from .semantic_scholar import (
    SemanticScholarError,
    SemanticScholarRateLimited,
    SemanticScholarRetriever,
)

__all__ = [
    "MAX_REFERENCE_QUERY_COUNT",
    "VALID_QUERY_CATEGORIES",
    "FallbackOutcome",
    "ReferencePlan",
    "ReferencePlanAnswer",
    "ReferenceQueryPlan",
    "ReferencesInput",
    "ReferencesOutput",
    "ReferencesResult",
    "SemanticScholarError",
    "SemanticScholarRateLimited",
    "SemanticScholarRetriever",
    "archive_round",
    "build_references_graph",
    "extract_bibtex_key",
    "load_bibtex_rows",
    "merge_reference_artifacts",
    "merge_reference_payloads",
    "record_reference_round",
    "reference_provenance_payload",
    "reference_query_plan_metadata",
    "reference_query_texts",
    "reference_repair_context",
    "retrieval_failure_message",
    "run_script_fallbacks",
    "validate_draft_reference_gaps_payload",
    "validate_reference_query_plan_payload",
    "write_draft_reference_gaps_validation_feedback",
    "write_plan_files",
    "write_reference_outputs",
    "write_reference_query_plan_validation_feedback",
]
