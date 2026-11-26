"""Evaluation of candidates and the sealed final evaluation, behind protocols."""

from ..core.errors import (
    EvaluatorOutputContractError,
    FinalEvaluationAlreadyStarted,
    FinalEvaluationError,
    FinalEvaluationInfrastructureError,
)
from .deferred import DeferredEvaluator, DeferredFinalEvaluator
from .evidence import (
    EVIDENCE_FILENAME,
    compute_usage,
    evidence_metadata,
    parse_evaluation_evidence,
    validate_compute_usage,
    validate_evaluator_outputs,
    validate_summaries,
)
from .ids import candidate_id, evaluation_id, final_id, worker_baseline_id
from .local import LocalCommandEvaluator, LocalFinalEvaluator, validate_final_manifest
from .program_clone import (
    clone_program,
    evaluation_results_complete,
    require_candidate_source_ready,
)
from .program_spec import (
    ProgramSpec,
    load_program_spec,
    parse_program_spec,
    render_default_program_spec_toml,
    resolve_entrypoint,
)
from .protocol import (
    ComputeUsage,
    EvaluationEvidence,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    EvidenceEpisode,
    EvidenceMetadata,
    FinalEvaluationRequest,
    FinalEvaluationResult,
    FinalEvaluator,
    FreezeRecord,
    SealedInputs,
)
from .reset_sequence import (
    declared_reset_sequence_matches,
    reduced_draft_reset_sequence_matches,
    reset_sequence_diagnostic,
)

__all__ = [
    "EVIDENCE_FILENAME",
    "ComputeUsage",
    "DeferredEvaluator",
    "DeferredFinalEvaluator",
    "EvaluationEvidence",
    "EvaluationRequest",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorOutputContractError",
    "EvidenceEpisode",
    "EvidenceMetadata",
    "FinalEvaluationAlreadyStarted",
    "FinalEvaluationError",
    "FinalEvaluationInfrastructureError",
    "FinalEvaluationRequest",
    "FinalEvaluationResult",
    "FinalEvaluator",
    "FreezeRecord",
    "LocalCommandEvaluator",
    "LocalFinalEvaluator",
    "ProgramSpec",
    "SealedInputs",
    "candidate_id",
    "clone_program",
    "compute_usage",
    "declared_reset_sequence_matches",
    "evaluation_id",
    "evaluation_results_complete",
    "evidence_metadata",
    "final_id",
    "load_program_spec",
    "parse_evaluation_evidence",
    "parse_program_spec",
    "reduced_draft_reset_sequence_matches",
    "render_default_program_spec_toml",
    "require_candidate_source_ready",
    "reset_sequence_diagnostic",
    "resolve_entrypoint",
    "validate_compute_usage",
    "validate_evaluator_outputs",
    "validate_final_manifest",
    "validate_summaries",
    "worker_baseline_id",
]
