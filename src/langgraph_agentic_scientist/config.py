"""Configuration of the method: every limit and switch of upstream, with upstream's defaults.

:class:`ScientistConfig` replaces upstream's Hydra configuration (``configs/config.yaml``,
``configs/variants_config.yaml``, ``configs/posthoc_config.yaml``, ``configs/paper_config.yaml``,
``configs/pipeline_config.yaml`` and the presets) and the constants of the shell drivers. Each
phase has its own section. Defaults are the stage defaults of upstream; the pipeline orchestrator
of upstream changes some of them, which :meth:`ScientistConfig.preset` reproduces.

Fields marked runtime-only (``json_schema_extra={"runtime_only": True}``: the ``agents`` section
and ``review.sampling``) are not part of the method. They are excluded from the configuration a
run records (:mod:`~langgraph_agentic_scientist.core.runconfig`) and may change when a run is
resumed. Every other field is the method: a resume with a different value raises
:class:`~langgraph_agentic_scientist.core.errors.ConfigurationMismatch`.

Unknown keys are refused, as Hydra refused them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUNTIME_ONLY: dict[str, Any] = {"runtime_only": True}

Phase = Literal[
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
]
"""The phases of the pipeline, in upstream's order."""

DEFAULT_PHASES: tuple[Phase, ...] = (
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
)
ALL_PHASES: tuple[Phase, ...] = (*DEFAULT_PHASES, "review", "revision")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DraftingConfig(_Section):
    """Task drafting (``task.sh``, ``lib/task_design.sh``, ``configs/config.yaml``)."""

    workflow: Literal["robust", "legacy"] = "robust"
    design_max_repairs: int = Field(8, ge=0)
    design_max_structure_repairs: int = Field(3, ge=0)
    design_review_attempts: int = Field(3, ge=1)
    design_format_attempts: int = Field(3, ge=1)
    design_delta_attempts: int = Field(3, ge=1)
    design_non_improving_before_focus: int = Field(2, ge=1)
    draft_max_attempts: int = Field(3, ge=1)
    contract_max_repairs: int = Field(3, ge=0)
    smoke_max_repairs: int = Field(3, ge=0)
    smoke_max_infrastructure_attempts: int = Field(3, ge=1)
    alignment_max_repairs: int = Field(10, ge=0)
    qualification_max_repairs: int = Field(4, ge=0, le=8)
    qualification_max_infrastructure_attempts: int = Field(3, ge=1)
    max_gate_checks_per_repair: int = Field(12, ge=1)
    evaluator_smoke_timeout_seconds: int = Field(1800, ge=1)
    production_entry_timeout_seconds: int = Field(300, ge=30, le=1800)
    qualification_timeout_seconds: int = Field(57600, ge=1)
    legacy_validation_max_repairs: int = Field(4, ge=0)
    max_runtime_bytes: int = Field(17_179_869_184, ge=1)


class InvestigationConfig(_Section):
    """Investigation (``configs/config.yaml`` ``execution``, ``run.sh``, ``idea.sh``)."""

    max_idea_cycles: int = Field(20, ge=1, le=1000)
    workers: int = Field(1, ge=1)
    max_program_evals_per_attempt: int = Field(4, ge=1)
    harness_evaluated_attempts: bool = False
    idea_timeout_seconds: int | None = Field(3600, ge=1)
    run_timeout_seconds: int | None = Field(21600, ge=1)
    harness_evaluation_timeout_seconds: int = Field(7200, ge=1)
    evaluation_timeout_seconds: int = Field(21600, ge=1)
    max_idea_generation_attempts: int = Field(2, ge=1)
    prune_every_n_cycles: int = Field(0, ge=0)
    parallel_fail_fast: bool = False
    isolate_worker_memory: bool = False
    max_harness_rubrics: int = Field(32, ge=1)
    max_source_patterns: int = Field(8, ge=1)
    max_source_pattern_chars: int = Field(1024, ge=1)
    validation_feedback_max_bytes: int = Field(2048, ge=1)
    harness_feedback_max_bytes: int = Field(8192, ge=1)
    public_contract_max_bytes: int = Field(16 * 1024, ge=1)


class VariantsConfig(_Section):
    """Ablations and baselines (``configs/variants_config.yaml``, ``variants_pipeline``)."""

    max_ablations: int = Field(3, ge=0)
    max_extra_baselines: int = Field(3, ge=0)
    max_evals_per_variant: int = Field(2, ge=1)
    edit_max_tries: int = Field(3, ge=1)
    verify_max_tries: int = Field(3, ge=1)
    num_replications: int = Field(1, ge=1)
    diff_max_chars: int = Field(50_000, ge=1)
    rubric_evidence_max_chars: int = Field(4096, ge=1)
    custom_only: bool = False
    custom_baselines: tuple[str, ...] = ()
    extra_tasks: tuple[str, ...] = ()
    baseline_program_override: str | None = None
    generation_timeout_seconds: int | None = Field(1800, ge=1)
    retrofit_timeout_seconds: int | None = Field(1800, ge=1)
    selection_timeout_seconds: int | None = Field(1800, ge=1)
    readme_timeout_seconds: int | None = Field(600, ge=1)
    variant_timeout_seconds: int | None = Field(7200, ge=1)
    cross_task_timeout_seconds: int | None = Field(7200, ge=1)
    evaluation_timeout_seconds: int | None = Field(None, ge=1)


class FinalEvaluationConfig(_Section):
    """Sealed final evaluation (``src/final_evaluation/runner.py``)."""

    timeout_seconds: int | None = Field(None, ge=1)


class PosthocConfig(_Section):
    """Post-hoc analysis (``configs/posthoc_config.yaml``, ``posthoc_pipeline``)."""

    bootstrap_max_attempts: int = Field(3, ge=1)
    max_selected_analyses: int = Field(3, ge=0)
    max_attempts_per_analysis: int = Field(3, ge=1)
    primary_metric_key: str | None = None
    primary_metric_tolerance: float = Field(0.01, ge=0)
    retrain_if_checkpoint_missing: bool = True
    checkpoint_mode: Literal["required", "optional"] = "required"
    evidence_max_files: int = Field(512, ge=1)
    evidence_max_file_bytes: int = Field(50 * 1024 * 1024, ge=1)
    evidence_max_total_bytes: int = Field(150 * 1024 * 1024, ge=1)
    bootstrap_timeout_seconds: int | None = Field(3600, ge=1)
    generation_timeout_seconds: int | None = Field(1800, ge=1)
    selection_timeout_seconds: int | None = Field(1800, ge=1)
    analysis_timeout_seconds: int | None = Field(3600, ge=1)


class PaperConfig(_Section):
    """Paper writing (``configs/paper_config.yaml``, ``paper.sh``, ``paper_edit.sh``)."""

    profile: Literal["default", "standalone"] = "default"
    two_stage: bool = False
    paper_format: str = "iclr"
    review_rounds: int = Field(0, ge=0)
    max_repair_rounds: int = Field(10, ge=0)
    timeout_seconds: int | None = Field(7200, ge=1)
    min_refs: int = Field(8, ge=0)
    max_refs: int = Field(15, ge=1)
    max_queries: int = Field(10, ge=1)
    watermark: bool | None = True
    concept_figure_full_min_seconds: int = Field(3600, ge=0)
    concept_figure_medium_min_seconds: int = Field(1800, ge=0)
    concept_figure_light_min_seconds: int = Field(300, ge=0)


class ReviewConfig(_Section):
    """AI review (``src/aireview.py``)."""

    num_reviewers: int = Field(5, ge=1)
    max_attempts: int = Field(2, ge=1)
    max_completion_tokens: int = Field(16384, ge=1)
    min_pdf_text_chars: int = Field(100, ge=0)
    fewshot: bool = True
    sampling: Literal["upstream", "none"] = Field("upstream", json_schema_extra=RUNTIME_ONLY)


class ModelRetryConfig(_Section):
    """Retries of transient provider failures inside an agent session (15/45/90 s upstream)."""

    max_retries: int = Field(3, ge=0)
    initial_delay: float = Field(15.0, ge=0)
    backoff_factor: float = Field(3.0, ge=0)
    max_delay: float = Field(90.0, ge=0)
    jitter: bool = False


class AgentsConfig(_Section):
    """Runtime settings of agent sessions; not part of the method (runtime-only).

    ``execute_timeout_seconds`` and ``execute_max_timeout_seconds`` are upstream's shell budget
    (``BASH_DEFAULT_TIMEOUT_MS=600000``, ``BASH_MAX_TIMEOUT_MS=7200000``). ``model_call_limit``
    bounds the model calls of one session; upstream has no such bound and the default is none.
    """

    execute_timeout_seconds: int = Field(600, ge=1)
    execute_max_timeout_seconds: int = Field(7200, ge=1)
    max_tool_output_bytes: int = Field(100_000, ge=1)
    model_call_limit: int | None = Field(None, ge=1)
    model_retry: ModelRetryConfig = Field(default_factory=ModelRetryConfig)


class ScientistConfig(_Section):
    """Every limit and switch of the method, with upstream's defaults.

    Attributes:
        phases: The phases a pipeline run executes (upstream ``execution.phases``).
        fail_fast: Stop at the first failed phase instead of degrading.
        no_winner_policy: ``best_available`` freezes a fallback program when investigation ends
            without a winner; ``skip`` skips the downstream phases.
        final_outcome_policy: Whether a verified ``not_qualified`` final evaluation may support a
            paper.
        allow_unisolated_contracted: Lift upstream's refusal to search or finally evaluate a
            contracted task with an evaluator that does not isolate candidates. For local
            experiments and tests; recorded in the run's configuration.
    """

    phases: tuple[Phase, ...] = DEFAULT_PHASES
    fail_fast: bool = True
    no_winner_policy: Literal["best_available", "skip"] = "best_available"
    final_outcome_policy: Literal["qualified_only", "allow_not_qualified"] = "qualified_only"
    allow_unisolated_contracted: bool = False
    drafting: DraftingConfig = Field(default_factory=DraftingConfig)
    investigation: InvestigationConfig = Field(default_factory=InvestigationConfig)
    variants: VariantsConfig = Field(default_factory=VariantsConfig)
    final_evaluation: FinalEvaluationConfig = Field(default_factory=FinalEvaluationConfig)
    posthoc: PosthocConfig = Field(default_factory=PosthocConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig, json_schema_extra=RUNTIME_ONLY)

    @classmethod
    def preset(cls, name: Literal["paper", "pipeline"]) -> ScientistConfig:
        """Return one of upstream's configurations of the pipeline orchestrator.

        ``"pipeline"`` is ``configs/pipeline_config.yaml``: 3 idea cycles, one evaluation per
        attempt, 5 replications and 5 evaluations per variant, 25-30 references and a standalone
        two-stage paper. ``"paper"`` is that file composed with ``configs/preset/paper.yaml``,
        upstream's campaign preset: all eight phases, 20 idea cycles, one idea generation attempt,
        two evaluations per attempt, a 3000 s idea timeout, harness-evaluated attempts, one
        replication and one evaluation per variant, the ICLR format and
        ``allow_not_qualified``.
        """
        pipeline = cls(
            investigation=InvestigationConfig(max_idea_cycles=3, max_program_evals_per_attempt=1),
            variants=VariantsConfig(num_replications=5, max_evals_per_variant=5),
            paper=PaperConfig(min_refs=25, max_refs=30, profile="standalone", two_stage=True),
        )
        if name == "pipeline":
            return pipeline
        if name == "paper":
            return pipeline.model_copy(
                update={
                    "phases": ALL_PHASES,
                    "final_outcome_policy": "allow_not_qualified",
                    "investigation": pipeline.investigation.model_copy(
                        update={
                            "max_idea_cycles": 20,
                            "max_idea_generation_attempts": 1,
                            "max_program_evals_per_attempt": 2,
                            "idea_timeout_seconds": 3000,
                            "run_timeout_seconds": 21600,
                            "harness_evaluated_attempts": True,
                        }
                    ),
                    "variants": pipeline.variants.model_copy(
                        update={
                            "num_replications": 1,
                            "max_evals_per_variant": 1,
                            "variant_timeout_seconds": 7200,
                            "cross_task_timeout_seconds": 7200,
                        }
                    ),
                    "paper": pipeline.paper.model_copy(update={"paper_format": "iclr"}),
                }
            )
        raise ValueError(f"unknown preset: {name!r}")

    def method_dump(self) -> dict[str, Any]:
        """Return the method part of the configuration as JSON (runtime-only fields excluded)."""
        return _method_part(self)


def _method_part(model: BaseModel) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, info in type(model).model_fields.items():
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get("runtime_only"):
            continue
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            result[name] = _method_part(value)
        else:
            result[name] = _jsonable(value)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def runtime_only_paths(model: type[BaseModel] = ScientistConfig, prefix: str = "") -> list[str]:
    """Return the dotted names of every runtime-only field of a configuration model."""
    paths: list[str] = []
    for name, info in model.model_fields.items():
        dotted = f"{prefix}{name}"
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get("runtime_only"):
            paths.append(dotted)
            continue
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths.extend(runtime_only_paths(annotation, f"{dotted}."))
    return paths


__all__ = [
    "ALL_PHASES",
    "DEFAULT_PHASES",
    "AgentsConfig",
    "DraftingConfig",
    "FinalEvaluationConfig",
    "InvestigationConfig",
    "ModelRetryConfig",
    "PaperConfig",
    "Phase",
    "PosthocConfig",
    "ReviewConfig",
    "ScientistConfig",
    "VariantsConfig",
    "runtime_only_paths",
]
