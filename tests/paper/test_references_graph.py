"""The references graph alone: planning with repair rounds, per-query retrieval, fallbacks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from langgraph_agentic_scientist import Hooks, LocalWorkspace, ScientistConfig, ScientistContext
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig, PaperConfig
from langgraph_agentic_scientist.core.interfaces import ReferenceQuery, ReferenceRecord
from langgraph_agentic_scientist.core.state import TaskRef
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.references import (
    FallbackOutcome,
    ReferencePlanAnswer,
    SemanticScholarRateLimited,
    build_references_graph,
)
from langgraph_agentic_scientist.paper.references import graph as graph_module
from langgraph_agentic_scientist.testing import FakeRetriever, ScriptedModels, structured_call
from langgraph_agentic_scientist.workspace import Workspace

TASK = TaskRef(root="task", goal="Pack circles.", rubrics="Be correct.")
INPUT: dict[str, Any] = {"task": TASK, "paper_dir": "paper"}


def fast_config(**updates: Any) -> ScientistConfig:
    """A configuration whose in-session model retries do not sleep."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, **updates)


def plan(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ready",
        "central_topic": "Packing circles",
        "queries": [
            {"query": "circle packing", "category": "core_method", "rationale": "method"},
            {"query": "packing search", "category": "baseline_comparator", "rationale": "base"},
            {"query": "evolution of programs", "category": "result_driven_theme", "rationale": "x"},
        ],
    }
    payload.update(updates)
    return payload


def answer(**updates: Any) -> Any:
    return structured_call(ReferencePlanAnswer, {"memo": "The memo.", "plan": plan(**updates)})


def record(key: str, *, source: str = "semantic_scholar") -> ReferenceRecord:
    return ReferenceRecord(
        key=key,
        title=key,
        bibtex=f"@article{{{key},\n  title={{{key}}}\n}}",
        source=source,
        provenance={"id": f"ref.s2.{key}", "paper_id": key, "citation_key": key},
    )


class Retriever(FakeRetriever):
    """A fake retriever that can fail and that offers a fallback."""

    def __init__(
        self,
        records: Mapping[str, Sequence[ReferenceRecord]] | None = None,
        *,
        errors: Mapping[str, Exception] | None = None,
        fallback_outcome: FallbackOutcome | None = None,
    ) -> None:
        super().__init__(records)
        self.errors = dict(errors or {})
        self.fallback_outcome = fallback_outcome
        self.fallback_calls: list[dict[str, Any]] = []
        self.attempts: dict[str, int] = {}

    async def search(self, query: ReferenceQuery, *, workspace: Workspace) -> list[ReferenceRecord]:
        self.attempts[query.query] = self.attempts.get(query.query, 0) + 1
        error = self.errors.get(query.query)
        if isinstance(error, SemanticScholarRateLimited) and self.attempts[query.query] > 2:
            error = None
        if error is not None:
            raise error
        return await super().search(query, workspace=workspace)

    async def fallback(self, queries: list[str], **kwargs: Any) -> FallbackOutcome:
        self.fallback_calls.append({"queries": queries, **kwargs})
        return self.fallback_outcome or FallbackOutcome()


@pytest.fixture
def context(
    workspace: LocalWorkspace, make_context: Callable[..., ScientistContext]
) -> Callable[..., ScientistContext]:
    def build(steps: Sequence[Any], retriever: Any = None, **paper: Any) -> ScientistContext:
        models = ScriptedModels.from_scripts(
            {Role.REFERENCE_PLANNER: list(steps)}, context_window=200_000
        )
        config = fast_config(paper=PaperConfig(**{"min_refs": 2, "max_refs": 3, **paper}))
        return make_context(models=models, retriever=retriever or Retriever(), config=config)

    return build


async def run(ctx: ScientistContext, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    app = build_references_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "refs"}}
    return await app.ainvoke(dict(inputs or INPUT), config, context=ctx)


def planner_prompts(ctx: ScientistContext) -> list[str]:
    model: Any = ctx.models.for_role(Role.REFERENCE_PLANNER)
    return [
        str(call[0].content) if len(call) == 1 else str(call[-1].content) for call in model.calls
    ]


def first_user_prompt(ctx: ScientistContext, index: int) -> str:
    model: Any = ctx.models.for_role(Role.REFERENCE_PLANNER)
    return next(m.content for m in model.calls[index] if m.type == "human")  # type: ignore[no-any-return]


async def test_a_plan_leads_to_references_until_the_minimum_is_reached(
    context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    retriever = Retriever(
        {
            "circle packing": [record("A"), record("B")],
            "packing search": [record("C")],
        }
    )
    ctx = context([answer()], retriever)
    result = (await run(ctx))["references"]
    assert result.status == "completed" and result.reference_count == 2
    assert result.sources == ["semantic_scholar"] and result.plan_repairs == 0
    assert result.bib_path == "paper/references.bib"
    # the first query found two references, the minimum: no further query was asked
    assert [query.query for query in retriever.queries] == ["circle packing"]
    assert retriever.queries[0].limit == 2
    assert retriever.queries[0].filters["search_limit"] == 25
    memo = (await workspace.read_bytes("paper/reference_query_plan.md")).decode()
    assert memo == "The memo."
    stored = json.loads((await workspace.read_bytes("paper/reference_query_plan.json")).decode())
    assert stored["queries"][0]["query"] == "circle packing"
    bib = (await workspace.read_bytes("paper/references.bib")).decode()
    assert "@article{A," in bib and "@article{B," in bib
    provenance = json.loads(
        (await workspace.read_bytes("paper/references_provenance.json")).decode()
    )
    assert provenance["count"] == 2 and provenance["queries"][0] == "circle packing"
    assert provenance["reference_query_plan"]["central_topic"] == "Packing circles"
    assert [row["id"] for row in provenance["references"]] == ["ref.s2.A", "ref.s2.B"]
    # the planner prompt is the upstream prompt filled with the task's goal and the paper paths
    prompt = first_user_prompt(ctx, 0)
    assert "Pack circles." in prompt and "Be correct." in prompt
    assert "`./reference_query_plan.json`" in prompt and "{{" not in prompt


async def test_later_queries_ask_only_for_what_is_missing(
    context: Callable[..., ScientistContext],
) -> None:
    retriever = Retriever({"circle packing": [record("A")], "packing search": [record("B")]})
    result = (await run(context([answer()], retriever)))["references"]
    assert result.reference_count == 2
    assert [(q.query, q.limit) for q in retriever.queries] == [
        ("circle packing", 2),
        ("packing search", 1),
    ]
    assert retriever.queries[1].filters["exclude_keys"] == ["A"]
    assert retriever.queries[1].filters["exclude_paper_ids"] == ["A"]


async def test_an_invalid_plan_is_repaired_with_the_diagnostic_context(
    context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = context(
        [answer(central_topic=""), answer()],
        Retriever({"circle packing": [record("A"), record("B")]}),
    )
    result = (await run(ctx))["references"]
    assert result.status == "completed" and result.plan_repairs == 1
    first, second = first_user_prompt(ctx, 0), first_user_prompt(ctx, 1)
    assert "# Repair the previous reference plan" not in first
    assert "# Repair the previous reference plan" in second
    assert "missing non-empty central_topic" in second  # the previous validation feedback, as data
    assert await workspace.exists(
        "paper/paper_diagnostics/reference_query_plan/round_0000/attempt_001/reference_query_plan.json"
    )
    assert await workspace.exists(
        "paper/paper_diagnostics/reference_query_plan/round_0001/attempt_001/diagnostic_state.json"
    )


async def test_the_same_invalid_plan_twice_ends_the_stage(
    context: Callable[..., ScientistContext],
) -> None:
    ctx = context([answer(queries=[]), answer(queries=[])])
    result = (await run(ctx))["references"]
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "reference_plan_repeated"
    assert "unchanged after repair" in result.failure.detail and result.plan_repairs == 1


async def test_exhausted_repair_rounds_end_the_stage(
    context: Callable[..., ScientistContext],
) -> None:
    steps = [answer(queries=[]), answer(central_topic="", queries=[]), answer(central_topic=" ")]
    result = (await run(context(steps, max_repair_rounds=1)))["references"]
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "reference_plan_exhausted"
    assert "exhausted reference query plan repair rounds after 1 attempts" in result.failure.detail


async def test_a_failed_plan_counts_as_a_validation_error_and_is_repaired(
    context: Callable[..., ScientistContext],
) -> None:
    failed = structured_call(
        ReferencePlanAnswer,
        {"memo": "m", "plan": {"status": "fail", "failure_reason": "nothing to search for"}},
    )
    ctx = context([failed, answer()], Retriever({"circle packing": [record("A"), record("B")]}))
    result = (await run(ctx))["references"]
    assert result.status == "completed" and result.plan_repairs == 1
    assert "reference query planning failed: nothing to search for" in first_user_prompt(ctx, 1)


async def test_preserve_mode_keeps_the_references_and_asks_no_model(
    context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    await workspace.write_bytes("paper/references.bib", b"@article{K,\n}\n\n@book{L,\n}\n")
    ctx = context([])
    result = (await run(ctx, {**INPUT, "reference_mode": "preserve"}))["references"]
    assert result.status == "preserved" and result.reference_count == 2
    model: Any = ctx.models.for_role(Role.REFERENCE_PLANNER)
    assert model.call_count == 0


async def test_the_edit_variant_uses_the_edit_prompt(
    context: Callable[..., ScientistContext],
) -> None:
    ctx = context([answer()], Retriever({"circle packing": [record("A"), record("B")]}))
    values = {"PAPER_EDIT_CONTEXT": "EDIT CONTEXT TEXT", "GLOBAL_MEMORY_RECENT_JSON": '[{"x": 1}]'}
    result = (await run(ctx, {**INPUT, "variant": "edit", "values": values}))["references"]
    assert result.status == "completed"
    prompt = first_user_prompt(ctx, 0)
    assert "# Edit Context\nEDIT CONTEXT TEXT" in prompt and '[{"x": 1}]' in prompt


async def test_a_failing_query_is_recorded_and_skipped(
    context: Callable[..., ScientistContext],
) -> None:
    retriever = Retriever(
        {"packing search": [record("A"), record("B")]},
        errors={"circle packing": ValueError("bad query")},
    )
    result = (await run(context([answer()], retriever)))["references"]
    assert result.status == "completed" and result.reference_count == 2
    assert result.retrieval_errors == ["ValueError: bad query"]


async def test_a_rate_limited_query_is_retried_by_the_node_policy(
    context: Callable[..., ScientistContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        graph_module,
        "RATE_LIMIT_RETRY",
        RetryPolicy(
            max_attempts=5,
            initial_interval=0.001,
            jitter=False,
            retry_on=SemanticScholarRateLimited,
        ),
    )
    retriever = Retriever(
        {"circle packing": [record("A"), record("B")]},
        errors={"circle packing": SemanticScholarRateLimited("Semantic Scholar HTTP 429 for u")},
    )
    result = (await run(context([answer()], retriever)))["references"]
    assert result.status == "completed" and retriever.attempts["circle packing"] == 3
    assert result.retrieval_errors == []


async def test_a_query_that_stays_rate_limited_is_skipped_after_the_retries(
    context: Callable[..., ScientistContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        graph_module,
        "RATE_LIMIT_RETRY",
        RetryPolicy(
            max_attempts=2,
            initial_interval=0.001,
            jitter=False,
            retry_on=SemanticScholarRateLimited,
        ),
    )

    class Always(Retriever):
        async def search(
            self, query: ReferenceQuery, *, workspace: Workspace
        ) -> list[ReferenceRecord]:
            if query.query == "circle packing":
                raise SemanticScholarRateLimited("Semantic Scholar HTTP 429 for u")
            return await FakeRetriever.search(self, query, workspace=workspace)

    retriever = Always({"packing search": [record("A"), record("B")]})
    result = (await run(context([answer()], retriever)))["references"]
    assert result.status == "completed"
    assert len(result.retrieval_errors) == 1 and "HTTP 429" in result.retrieval_errors[0]


async def test_a_forbidden_key_skips_to_the_fallbacks(
    context: Callable[..., ScientistContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        graph_module,
        "RATE_LIMIT_RETRY",
        RetryPolicy(
            max_attempts=1,
            initial_interval=0.001,
            jitter=False,
            retry_on=SemanticScholarRateLimited,
        ),
    )
    outcome = FallbackOutcome(
        records=[record("O1", source="openalex"), record("O2", source="openalex")],
        sources=["openalex"],
    )
    retriever = Retriever(
        errors={"circle packing": SemanticScholarRateLimited("Semantic Scholar HTTP 403 for u")},
        fallback_outcome=outcome,
    )
    result = (await run(context([answer()], retriever)))["references"]
    assert result.status == "completed" and result.sources == ["openalex"]
    assert [query.query for query in retriever.queries] == []  # no other query was tried
    (call,) = retriever.fallback_calls
    assert call["min_refs"] == 2 and call["max_refs"] == 3 and call["s2_count"] == 0
    assert "did not meet minimum references: 0 < 2" in call["s2_error"]
    assert call["queries"] == ["circle packing", "packing search", "evolution of programs"]


async def test_too_few_references_go_to_the_fallbacks_and_their_failure_is_reported(
    context: Callable[..., ScientistContext],
) -> None:
    outcome = FallbackOutcome(openalex_error="oa down", scholarly_error="sc down")
    retriever = Retriever({"circle packing": [record("A")]}, fallback_outcome=outcome)
    result = (await run(context([answer()], retriever)))["references"]
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "reference_retrieval_failed"
    assert result.failure.detail.startswith("Reference retrieval failed. Semantic Scholar: ")
    assert "OpenAlex: oa down; scholarly: sc down" in result.failure.detail
    assert len(retriever.queries) == 3  # every query was tried before the fallbacks


async def test_a_retriever_without_fallbacks_fails_with_its_own_error(
    context: Callable[..., ScientistContext],
) -> None:
    result = (await run(context([answer()], FakeRetriever())))["references"]
    assert result.status == "failed" and result.failure is not None
    assert result.failure.detail.startswith("Semantic Scholar retrieval did not meet minimum")


async def test_hooks_run_around_the_steps_of_the_graph(
    context: Callable[..., ScientistContext], make_context: Callable[..., ScientistContext]
) -> None:
    seen: list[str] = []

    async def before_plan(state: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
        seen.append("before plan")
        return {"hook_outputs": {"note": "x"}}

    base = context([answer()], Retriever({"circle packing": [record("A"), record("B")]}))
    ctx = ScientistContext(
        models=base.models, workspace=base.workspace, retriever=base.retriever, config=base.config,
        hooks=Hooks(before={"references.plan": [before_plan]}),
    )  # fmt: skip
    output = await run(ctx)
    assert seen == ["before plan"] and output["references"].status == "completed"


async def test_a_crash_after_retrieval_resumes_without_repeating_a_step(
    context: Callable[..., ScientistContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = Retriever({"circle packing": [record("A")], "packing search": [record("B")]})
    ctx = context([answer()], retriever)
    real = graph_module.write_reference_outputs
    crashes: list[int] = []

    async def crash_once(*args: Any, **kwargs: Any) -> None:
        if not crashes:
            crashes.append(1)
            raise ConnectionResetError("the workspace went away")
        await real(*args, **kwargs)

    monkeypatch.setattr(graph_module, "write_reference_outputs", crash_once)
    app = build_references_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume"}}
    with pytest.raises(ConnectionResetError):
        await app.ainvoke(dict(INPUT), config, context=ctx)
    output = await app.ainvoke(None, config, context=ctx)
    assert output["references"].reference_count == 2
    model: Any = ctx.models.for_role(Role.REFERENCE_PLANNER)
    assert model.call_count == 1  # the planner ran once
    assert [query.query for query in retriever.queries] == ["circle packing", "packing search"]
