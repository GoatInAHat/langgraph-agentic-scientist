"""A LangGraph port of Sakana AI's Agentic Scientist (AI-Scientist-v3).

The core exports below are what every phase package builds on: configuration, the run context,
models by role, prompts, workspaces, evaluators and the shared records. ``build_pipeline_graph``
chains the phases. Every phase has a factory that returns a compiled graph, usable alone or
mounted in a larger graph; the factories and the phase packages' own classes are exported here as
well as from their subpackages::

    from langgraph_agentic_scientist import build_paper_graph, build_pipeline_graph
"""

from __future__ import annotations

from .config import (
    DraftingConfig,
    FinalEvaluationConfig,
    InvestigationConfig,
    PaperConfig,
    PosthocConfig,
    ReviewConfig,
    ScientistConfig,
    VariantsConfig,
)
from .context import (
    AutoApprovePolicy,
    GraphOverrides,
    Hooks,
    HumanPolicy,
    InterruptPolicy,
    ScientistContext,
    StepHook,
)
from .core.contract import TaskContract, load_task_contract
from .core.errors import (
    ConfigurationMismatch,
    FinalEvaluationError,
    IncompatibleCheckpoint,
    ProgramSpecError,
    TaskContractError,
)
from .core.interfaces import (
    CompileResult,
    ConceptFigureRenderer,
    PaperToolchain,
    ReferenceQuery,
    ReferenceRecord,
    ReferenceRetriever,
    RubricVerifier,
    TaskGate,
)
from .core.results import (
    AIReview,
    AnalysisResult,
    AttemptRecord,
    DownstreamSelection,
    DraftingResult,
    EditOptions,
    GroundingSource,
    IdeaRecord,
    InvestigationResult,
    PaperData,
    PaperResult,
    PosthocResult,
    ReviewResult,
    RevisionResult,
    ValidationReport,
    VariantResult,
    VariantSpec,
    VariantsResult,
)
from .core.state import BestManifest, ProgramRef, TaskRef
from .drafting import LocalTaskGate, build_task_design_graph, build_task_drafting_graph
from .evaluation import (
    DeferredEvaluator,
    DeferredFinalEvaluator,
    EvaluationEvidence,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    FinalEvaluationAlreadyStarted,
    FinalEvaluationRequest,
    FinalEvaluationResult,
    FinalEvaluator,
    LocalCommandEvaluator,
    LocalFinalEvaluator,
    SealedInputs,
)
from .final_evaluation import (
    DRAW_INPUTS_SLOT,
    DrawInputsInput,
    DrawInputsOutput,
    build_final_evaluation_graph,
)
from .investigation import (
    HarnessRubricVerifier,
    build_attempt_graph,
    build_campaign_worker_graph,
    build_downstream_selection_graph,
    build_ideation_graph,
    build_investigation_graph,
)
from .models import ModelRegistry, Role
from .paper import (
    AutoillustrateRenderer,
    SemanticScholarRetriever,
    WorkspacePaperToolchain,
    build_paper_graph,
    build_references_graph,
)
from .pipeline import (
    TASK_SNAPSHOT_DIR,
    PipelineConfigurationError,
    PipelineInput,
    PipelineOutput,
    PipelineState,
    build_pipeline_graph,
    load_task_ref,
    prepare_task_snapshot,
    reseal_task_snapshot,
)
from .posthoc import build_posthoc_graph
from .prompts import PromptRegistry
from .review import build_review_graph
from .revision import build_revision_graph
from .serde import serde_allowlist
from .variants import build_variants_graph
from .workspace import CommandResult, LocalWorkspace, SnapshotRef, Workspace, WorkspaceBase

__all__ = [
    "DRAW_INPUTS_SLOT",
    "TASK_SNAPSHOT_DIR",
    "AIReview",
    "AnalysisResult",
    "AttemptRecord",
    "AutoApprovePolicy",
    "AutoillustrateRenderer",
    "BestManifest",
    "CommandResult",
    "CompileResult",
    "ConceptFigureRenderer",
    "ConfigurationMismatch",
    "DeferredEvaluator",
    "DeferredFinalEvaluator",
    "DownstreamSelection",
    "DraftingConfig",
    "DraftingResult",
    "DrawInputsInput",
    "DrawInputsOutput",
    "EditOptions",
    "EvaluationEvidence",
    "EvaluationRequest",
    "EvaluationResult",
    "Evaluator",
    "FinalEvaluationAlreadyStarted",
    "FinalEvaluationConfig",
    "FinalEvaluationError",
    "FinalEvaluationRequest",
    "FinalEvaluationResult",
    "FinalEvaluator",
    "GraphOverrides",
    "GroundingSource",
    "HarnessRubricVerifier",
    "Hooks",
    "HumanPolicy",
    "IdeaRecord",
    "IncompatibleCheckpoint",
    "InterruptPolicy",
    "InvestigationConfig",
    "InvestigationResult",
    "LocalCommandEvaluator",
    "LocalFinalEvaluator",
    "LocalTaskGate",
    "LocalWorkspace",
    "ModelRegistry",
    "PaperConfig",
    "PaperData",
    "PaperResult",
    "PaperToolchain",
    "PipelineConfigurationError",
    "PipelineInput",
    "PipelineOutput",
    "PipelineState",
    "PosthocConfig",
    "PosthocResult",
    "ProgramRef",
    "ProgramSpecError",
    "PromptRegistry",
    "ReferenceQuery",
    "ReferenceRecord",
    "ReferenceRetriever",
    "ReviewConfig",
    "ReviewResult",
    "RevisionResult",
    "Role",
    "RubricVerifier",
    "ScientistConfig",
    "ScientistContext",
    "SealedInputs",
    "SemanticScholarRetriever",
    "SnapshotRef",
    "StepHook",
    "TaskContract",
    "TaskContractError",
    "TaskGate",
    "TaskRef",
    "ValidationReport",
    "VariantResult",
    "VariantSpec",
    "VariantsConfig",
    "VariantsResult",
    "Workspace",
    "WorkspaceBase",
    "WorkspacePaperToolchain",
    "build_attempt_graph",
    "build_campaign_worker_graph",
    "build_downstream_selection_graph",
    "build_final_evaluation_graph",
    "build_ideation_graph",
    "build_investigation_graph",
    "build_paper_graph",
    "build_pipeline_graph",
    "build_posthoc_graph",
    "build_references_graph",
    "build_review_graph",
    "build_revision_graph",
    "build_task_design_graph",
    "build_task_drafting_graph",
    "build_variants_graph",
    "load_task_contract",
    "load_task_ref",
    "prepare_task_snapshot",
    "reseal_task_snapshot",
    "serde_allowlist",
]
