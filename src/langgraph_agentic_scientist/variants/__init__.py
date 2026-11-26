"""Ablations and baselines: the variants phase of the Agentic Scientist (upstream's
``variants_pipeline``, stages 0-8).

Given a completed investigation, the phase extracts the best program, retrofits it to emit
analysis CSVs while the evaluator stays sealed, has agents propose and filter baselines and
ablations, materialises one workspace per variant, runs the control first and the other variants
after it (each an edit, a host evaluation and a rubric verification), adapts every variant to
optional extra tasks, aggregates the CSVs and writes a README for the paper stage.

:func:`build_variants_graph` returns the compiled graph; :func:`build_run_variant_graph` and
:func:`build_cross_task_graph` return its parts, each usable alone.
"""

from .analysis import write_canonical_analysis
from .cross_task import CrossTaskInput, CrossTaskOutput, build_cross_task_graph
from .execute import (
    RunVariantInput,
    RunVariantOutput,
    build_run_variant_graph,
    run_variant_input,
)
from .graph import VariantsInput, VariantsOutput, VariantsState, build_variants_graph
from .models import (
    BundleRef,
    RetrofitReport,
    VariantBundleAnswer,
    VariantCandidate,
    VariantDecisionsAnswer,
    VariantRun,
    VariantSelection,
    VariantSummariesAnswer,
)
from .rubrics import build_rubric_contract, required_rubric_ids

__all__ = [
    "BundleRef",
    "CrossTaskInput",
    "CrossTaskOutput",
    "RetrofitReport",
    "RunVariantInput",
    "RunVariantOutput",
    "VariantBundleAnswer",
    "VariantCandidate",
    "VariantDecisionsAnswer",
    "VariantRun",
    "VariantSelection",
    "VariantSummariesAnswer",
    "VariantsInput",
    "VariantsOutput",
    "VariantsState",
    "build_cross_task_graph",
    "build_rubric_contract",
    "build_run_variant_graph",
    "build_variants_graph",
    "required_rubric_ids",
    "run_variant_input",
    "write_canonical_analysis",
]
