"""The references graph: plan the literature search, retrieve references, write the bibliography.

``build_references_graph`` is the paper phase's step 2 (upstream ``paper.sh``, phase ``initial``,
and its edit-mode counterpart). It runs alone or mounted as the ``paper.references`` slot:

``record_run_config`` -> (preserve mode: ``preserve`` and stop) -> ``plan`` -> ``check_plan`` ->
(an invalid plan: ``plan`` again, up to ``paper.max_repair_rounds`` repair rounds) ->
``retrieve`` (one node execution per query, until ``paper.min_refs`` references are found) ->
(too few: ``fallback``) -> ``write_bibliography``.

* ``plan`` runs the reference planner (role R27), which answers with a memo and a plan through
  structured output. In a repair round the prompt carries the repair context of upstream's
  ``paper_diagnostics`` (the allowed categories and the previous artifacts, as data). The
  harness writes ``reference_query_plan.md`` and ``.json`` from the answer.
* ``check_plan`` validates the plan files (upstream's rules), writes the validation feedback,
  archives the round under ``paper_diagnostics/``, and stops the stage when two consecutive
  rounds repeat the same invalid plan and errors, or the repair rounds are exhausted.
* ``retrieve`` asks the context's :class:`~langgraph_agentic_scientist.ReferenceRetriever`
  (default: Semantic Scholar) for the references of one query. A rate-limited response is
  retried by the node's ``RetryPolicy``; a query that still fails is recorded and skipped.
* ``fallback`` runs when the retriever offers ``fallback`` (the default one runs upstream's
  OpenAlex and scholarly scripts) and the queries found too few references.
* ``write_bibliography`` writes ``references.bib`` and ``references_provenance.json``, or fails
  with upstream's message.

The graph's input and output are :class:`ReferencesInput` and :class:`ReferencesOutput`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, NotRequired, Protocol, TypedDict, runtime_checkable

from langgraph.errors import NodeError
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command, RetryPolicy
from pydantic import Field

from ...agents.runner import agent_node_retry_policy, run_structured_role
from ...context import GraphOverrides, ScientistContext
from ...core.interfaces import ReferenceQuery, ReferenceRecord, ReferenceRetriever
from ...core.runconfig import method_config, record_run_config
from ...core.state import Failure, RunConfigRecord, StateModel, TaskRef, merge_dicts, utc_now
from ...core.steps import StepGraph
from ...models import Role
from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .bibtex import load_bibtex_rows
from .diagnostics import record_reference_round, reference_repair_context
from .plan import (
    ReferencePlan,
    ReferencePlanAnswer,
    reference_query_plan_metadata,
    reference_query_texts,
    write_plan_files,
    write_reference_query_plan_validation_feedback,
)
from .retrieve import (
    FallbackOutcome,
    reference_provenance_payload,
    retrieval_failure_message,
    write_reference_outputs,
)
from .semantic_scholar import SemanticScholarRateLimited, SemanticScholarRetriever

PLAN_STAGE = "reference_query_plan"
"""The stem of the plan files and the diagnostics stage (``reference_query_plan.md`` ...)."""

PLAN_PROMPT_KEYS = {
    "initial": "paper/reference_query_plan",
    "edit": "paper/reference_query_plan_edit",
}
"""The registry keys of the planner prompt by variant."""

PAPER_PATH_VALUES: dict[str, str] = {
    "PAPER_DIR": ".",
    "PAPER_DATA_PATH": "./paper_data.json",
    "PAPER_BIB_PATH": "./references.bib",
    "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
    "PAPER_FIGURES_DIR": "./figures",
    "PAPER_TEX_PATH": "./paper.tex",
    "PAPER_CLAIMS_PATH": "./claims_grounding.json",
    "REFERENCE_QUERY_PLAN_PATH": "./reference_query_plan.md",
    "REFERENCE_QUERY_PLAN_JSON_PATH": "./reference_query_plan.json",
    "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    "GLOBAL_MEMORY_MD_FILE": "./global_memory/memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./global_memory/memory.jsonl",
    "PAPER_EDIT_CONTEXT_PATH": "./paper_edit_context.md",
    "PAPER_EDIT_CONTEXT_JSON_PATH": "./paper_edit_context.json",
}
"""Upstream's path placeholders of the paper workspace (the ``extra`` table of ``paper.sh``)."""

SEARCH_LIMIT_PER_QUERY = 25
"""Candidates requested per search (upstream ``search_limit_per_query``)."""

RATE_LIMIT_RETRY = RetryPolicy(
    max_attempts=5,
    initial_interval=1.5,
    backoff_factor=2.0,
    max_interval=20.0,
    jitter=False,
    retry_on=SemanticScholarRateLimited,
)
"""Retries of a rate-limited query (upstream retried each request five times, up to 20 s apart)."""


class ReferencesResult(StateModel):
    """What the references graph returns.

    Attributes:
        status: ``completed``, ``preserved`` (preserve mode: the previous references stay) or
            ``failed``.
        plan: The validated query plan, when planning succeeded.
        plan_repairs: The repair rounds the planner needed.
        reference_count: The references written.
        bib_path: The workspace path of ``references.bib``.
        provenance_path: The workspace path of ``references_provenance.json``.
        sources: The providers the references came from (``semantic_scholar``, ``openalex``,
            ``scholarly``, or the caller's).
        retrieval_errors: Errors of queries that failed and were skipped.
        failure: Why the stage failed.
    """

    status: Literal["completed", "preserved", "failed"]
    plan: ReferencePlan | None = None
    plan_repairs: int = 0
    reference_count: int = 0
    bib_path: str | None = None
    provenance_path: str | None = None
    sources: list[str] = Field(default_factory=list)
    retrieval_errors: list[str] = Field(default_factory=list)
    failure: Failure | None = None


class FallbackRecord(StateModel):
    """The checkpointed outcome of the script fallbacks."""

    records: list[ReferenceRecord] = Field(default_factory=list)
    openalex_error: str | None = None
    scholarly_error: str | None = None
    sources: list[str] = Field(default_factory=list)


class ReferencesInput(TypedDict):
    """What the references graph is given.

    ``values`` holds the placeholder values of the planner prompt that the caller computes
    (``TASK_GOAL``, ``TASK_RUBRICS``, ``GLOBAL_MEMORY_RECENT_JSON``, and in edit mode
    ``PAPER_EDIT_CONTEXT``); the standard path placeholders of the paper workspace and the task's
    goal and rubrics are filled when missing.
    """

    task: TaskRef
    paper_dir: str
    run_config: NotRequired[RunConfigRecord | None]
    reference_mode: NotRequired[Literal["refresh", "preserve"]]
    variant: NotRequired[Literal["initial", "edit"]]
    values: NotRequired[dict[str, str]]
    results_dir: NotRequired[str]
    campaign_results_dir: NotRequired[str]
    generated_at: NotRequired[str]


class ReferencesOutput(TypedDict):
    """What the references graph returns: ``references``."""

    references: ReferencesResult


class ReferencesState(ReferencesInput):
    """The state of the references graph."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    plan: ReferencePlan | None
    plan_repairs: int
    plan_failure: str | None
    queries: list[str]
    query_index: int
    records: list[ReferenceRecord]
    retrieval_errors: list[str]
    s2_error: str | None
    s2_disabled: bool
    fallback_outcome: FallbackRecord | None
    references: ReferencesResult | None


@runtime_checkable
class FallbackRetriever(Protocol):
    """A retriever that also offers fallbacks for a retrieval that fell short.

    :class:`SemanticScholarRetriever` is one. The graph calls ``fallback`` after the per-query
    retrieval when fewer than ``paper.min_refs`` references were found.
    """

    async def fallback(
        self,
        queries: list[str],
        *,
        min_refs: int,
        max_refs: int,
        paper_dir: str,
        workspace: Workspace,
        s2_error: str | None,
        s2_count: int,
    ) -> FallbackOutcome:
        """Return the references of the fallbacks, or the reasons they failed."""
        ...


def _retriever(runtime: Runtime[ScientistContext]) -> ReferenceRetriever:
    return runtime.context.retriever or SemanticScholarRetriever()


def planner_values(state: Mapping[str, Any]) -> dict[str, str]:
    """Return the placeholder values of the planner prompt for a state of the graph."""
    task: TaskRef = state["task"]
    values: dict[str, str] = {
        "TASK_GOAL": task.goal,
        "TASK_RUBRICS": task.rubrics,
        "GLOBAL_MEMORY_RECENT_JSON": "[]",
        "PAPER_EDIT_CONTEXT": "",
        **PAPER_PATH_VALUES,
    }
    values.update(state.get("values") or {})
    return values


# --- nodes ----------------------------------------------------------------------------------


def _route_mode(state: ReferencesState) -> str:
    return "preserve" if state.get("reference_mode", "refresh") == "preserve" else "plan"


async def _preserve(state: ReferencesState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    workspace = runtime.context.workspace
    paper_dir = state["paper_dir"]
    bib = join(paper_dir, "references.bib")
    count = 0
    if await workspace.exists(bib):
        count = len(load_bibtex_rows((await workspace.read_bytes(bib)).decode("utf-8")))
    return {
        "references": ReferencesResult(
            status="preserved",
            reference_count=count,
            bib_path=bib,
            provenance_path=join(paper_dir, "references_provenance.json"),
        )
    }


async def _plan(state: ReferencesState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    config = method_config(state)
    variant = state.get("variant", "initial")
    paper_dir = state["paper_dir"]
    repairs = state.get("plan_repairs", 0)
    suffix = ""
    if repairs > 0:
        suffix = await reference_repair_context(
            ctx.workspace, paper_dir, PLAN_STAGE, prompts=ctx.prompts
        )
    outcome = await run_structured_role(
        role=Role.REFERENCE_PLANNER,
        schema=ReferencePlanAnswer,
        runtime=runtime,
        root=paper_dir,
        read_only=True,
        prompt_key=PLAN_PROMPT_KEYS[variant],
        values=planner_values(state),
        suffix=suffix,
        timeout=config.paper.timeout_seconds,
    )
    answer = outcome.structured
    if isinstance(answer, ReferencePlanAnswer):
        await write_plan_files(
            ctx.workspace,
            paper_dir,
            stem=PLAN_STAGE,
            memo=answer.memo,
            plan=answer.plan.model_dump(mode="json"),
        )
    return {}


async def _check_plan(state: ReferencesState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    config = method_config(state)
    paper_dir = state["paper_dir"]
    repairs = state.get("plan_repairs", 0)
    feedback = await write_reference_query_plan_validation_feedback(
        ctx.workspace, paper_dir, stem=PLAN_STAGE, max_queries=config.paper.max_queries
    )
    repeated = await record_reference_round(ctx.workspace, paper_dir, PLAN_STAGE, repairs)
    if feedback.get("ok"):
        payload = json.loads(
            (await ctx.workspace.read_bytes(join(paper_dir, f"{PLAN_STAGE}.json"))).decode("utf-8")
        )
        return {
            "plan": ReferencePlan.model_validate(_plan_fields(payload)),
            "plan_failure": None,
            "queries": reference_query_texts(payload),
            "query_index": 0,
            "records": [],
            "retrieval_errors": [],
            "s2_error": None,
            "s2_disabled": False,
        }
    errors = "; ".join(str(message) for message in feedback.get("hard_errors", []))
    if repeated:
        return {
            "plan_failure": "reference_plan_repeated",
            "references": ReferencesResult(
                status="failed",
                plan_repairs=repairs,
                failure=Failure(
                    category="reference_plan_repeated",
                    scope="method",
                    detail=(
                        "Reference plan and validation errors unchanged after repair: " + errors
                    ),
                ),
            ),
        }
    if repairs >= config.paper.max_repair_rounds:
        return {
            "plan_failure": "reference_plan_exhausted",
            "references": ReferencesResult(
                status="failed",
                plan_repairs=repairs,
                failure=Failure(
                    category="reference_plan_exhausted",
                    scope="method",
                    detail=(
                        "exhausted reference query plan repair rounds after "
                        f"{repairs} attempts: {errors}"
                    ),
                ),
            ),
        }
    return {"plan_repairs": repairs + 1, "plan_failure": "invalid"}


def _plan_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    known = set(ReferencePlan.model_fields)
    fields = {key: value for key, value in payload.items() if key in known}
    queries = fields.get("queries")
    if isinstance(queries, list):
        fields["queries"] = [
            {k: v for k, v in row.items() if k in {"query", "category", "rationale"}}
            for row in queries
            if isinstance(row, dict)
        ]
    return fields


def _after_check(state: ReferencesState) -> str:
    failure = state.get("plan_failure")
    if failure is None:
        return "retrieve"
    if failure == "invalid":
        return "plan"
    return "done"


async def _retrieve(state: ReferencesState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    config = method_config(state)
    records: list[ReferenceRecord] = list(state.get("records", []))
    index = state["query_index"]
    remaining = max(config.paper.min_refs - len(records), 1)
    query = ReferenceQuery(
        query=state["queries"][index],
        limit=remaining,
        filters={
            "search_limit": SEARCH_LIMIT_PER_QUERY,
            "exclude_paper_ids": [
                str(record.provenance["paper_id"])
                for record in records
                if "paper_id" in record.provenance
            ],
            "exclude_keys": [record.key for record in records],
        },
    )
    found = await _retriever(runtime).search(query, workspace=ctx.workspace)
    keys = {record.key for record in records}
    for record in found:
        if record.key not in keys:
            keys.add(record.key)
            records.append(record)
    return {"records": records, "query_index": index + 1}


async def _retrieve_failed(
    state: ReferencesState, error: NodeError
) -> Command[Literal["after_query"]]:
    message = f"{type(error.error).__name__}: {error.error}"
    errors = [*state.get("retrieval_errors", []), message]
    update: dict[str, Any] = {
        "query_index": state["query_index"] + 1,
        "retrieval_errors": errors,
    }
    if isinstance(error.error, SemanticScholarRateLimited) and "HTTP 403" in str(error.error):
        update["s2_disabled"] = True
    return Command(update=update, goto="after_query")


def _s2_error_text(state: ReferencesState, min_refs: int) -> str | None:
    records = state.get("records", [])
    if len(records) >= min_refs:
        return None
    text = (
        f"Semantic Scholar retrieval did not meet minimum references: {len(records)} < {min_refs}"
    )
    errors = state.get("retrieval_errors", [])
    if errors:
        text += f"; last_error={errors[-1]}"
    return text


async def _decide(state: ReferencesState) -> dict[str, Any]:
    config = method_config(state)
    return {"s2_error": _s2_error_text(state, config.paper.min_refs)}


def _after_decide(state: ReferencesState) -> str:
    config = method_config(state)
    records = state.get("records", [])
    if len(records) >= config.paper.min_refs:
        return "write_bibliography"
    if state.get("s2_disabled") or state["query_index"] >= len(state["queries"]):
        return "fallback"
    return "retrieve"


async def _fallback(state: ReferencesState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    config = method_config(state)
    retriever = _retriever(runtime)
    if not isinstance(retriever, FallbackRetriever):
        return {"fallback_outcome": FallbackRecord()}
    outcome = await retriever.fallback(
        list(state["queries"]),
        min_refs=config.paper.min_refs,
        max_refs=config.paper.max_refs,
        paper_dir=state["paper_dir"],
        workspace=runtime.context.workspace,
        s2_error=state.get("s2_error"),
        s2_count=len(state.get("records", [])),
    )
    return {
        "fallback_outcome": FallbackRecord(
            records=outcome.records,
            openalex_error=outcome.openalex_error,
            scholarly_error=outcome.scholarly_error,
            sources=outcome.sources,
        )
    }


async def _write_bibliography(
    state: ReferencesState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    ctx = runtime.context
    config = method_config(state)
    paper_dir = state["paper_dir"]
    plan = state.get("plan")
    records = list(state.get("records", []))
    fallback = state.get("fallback_outcome")
    chosen: list[ReferenceRecord]
    if fallback is not None and fallback.records:
        chosen = list(fallback.records)
    elif len(records) >= config.paper.min_refs:
        chosen = records
    else:
        message = retrieval_failure_message(
            state.get("s2_error"),
            fallback.openalex_error if fallback else None,
            fallback.scholarly_error if fallback else None,
        )
        return {
            "references": ReferencesResult(
                status="failed",
                plan=plan,
                plan_repairs=state.get("plan_repairs", 0),
                retrieval_errors=list(state.get("retrieval_errors", [])),
                failure=Failure(
                    category="reference_retrieval_failed", scope="infrastructure", detail=message
                ),
            )
        }
    payload = reference_provenance_payload(
        chosen,
        generated_at=state.get("generated_at") or utc_now(),
        results_dir=state.get("results_dir", ""),
        campaign_results_dir=state.get("campaign_results_dir", ""),
        queries=list(state["queries"]),
        plan_metadata=reference_query_plan_metadata(plan.model_dump(mode="json") if plan else {}),
    )
    await write_reference_outputs(ctx.workspace, paper_dir, chosen, payload)
    sources: list[str] = []
    for record in chosen:
        if record.source and record.source not in sources:
            sources.append(record.source)
    return {
        "references": ReferencesResult(
            status="completed",
            plan=plan,
            plan_repairs=state.get("plan_repairs", 0),
            reference_count=len(chosen),
            bib_path=join(paper_dir, "references.bib"),
            provenance_path=join(paper_dir, "references_provenance.json"),
            sources=sources,
            retrieval_errors=list(state.get("retrieval_errors", [])),
        )
    }


def build_references_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "references",
) -> CompiledStateGraph[ReferencesState, ScientistContext, ReferencesInput, ReferencesOutput]:
    """Build the references graph (planning, retrieval and the bibliography).

    Args:
        checkpointer: The caller's checkpointer (``None`` when the graph is mounted in a graph
            that has one).
        store: The caller's store (unused by this graph; accepted for the factory shape).
        overrides: Replacement slots (the steps of this graph have hooks, named
            ``references.<step>``).
        name: The compiled graph's name.

    Returns:
        A compiled graph with :class:`ReferencesInput` as input and :class:`ReferencesOutput` as
        output.
    """
    graph = StepGraph(
        "references",
        ReferencesState,
        input_schema=ReferencesInput,
        output_schema=ReferencesOutput,
        overrides=overrides,
    )
    graph.add_node("record_run_config", record_run_config)
    graph.add_node("preserve", _preserve)
    graph.step("plan", _plan, retry_policy=agent_node_retry_policy())
    graph.step("check_plan", _check_plan)
    graph.step("retrieve", _retrieve, retry_policy=RATE_LIMIT_RETRY, error_handler=_retrieve_failed)
    graph.add_node("after_query", _decide)
    graph.step("fallback", _fallback)
    graph.step("write_bibliography", _write_bibliography)

    graph.graph.add_edge(START, "record_run_config")
    graph.graph.add_conditional_edges(
        "record_run_config", _route_mode, {"preserve": "preserve", "plan": graph.entry("plan")}
    )
    graph.graph.add_edge("preserve", END)
    graph.add_edge("plan", "check_plan")
    graph.add_conditional_edges(
        "check_plan", _after_check, {"plan": "plan", "retrieve": "retrieve", "done": END}
    )
    graph.add_edge("retrieve", "after_query")
    graph.graph.add_conditional_edges(
        "after_query",
        _after_decide,
        {
            "retrieve": graph.entry("retrieve"),
            "fallback": graph.entry("fallback"),
            "write_bibliography": graph.entry("write_bibliography"),
        },
    )
    graph.add_edge("fallback", "write_bibliography")
    graph.add_edge("write_bibliography", END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "PAPER_PATH_VALUES",
    "PLAN_PROMPT_KEYS",
    "PLAN_STAGE",
    "FallbackRecord",
    "FallbackRetriever",
    "ReferencesInput",
    "ReferencesOutput",
    "ReferencesResult",
    "ReferencesState",
    "build_references_graph",
    "planner_values",
]
