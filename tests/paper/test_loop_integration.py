"""The loop with the host side's default slots: the references graph and the post-draft
enrichment's planning, retrieval and merge (no stub slots)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from test_loop_support import (
    audit_pass,
    fast_config,
    models,
    no_gaps,
    paper_config,
    reports,
    run,
    setup_task,
    write_paper,
)

from langgraph_agentic_scientist.core.interfaces import ReferenceRecord
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper import build_paper_graph
from langgraph_agentic_scientist.paper.models import DraftGapsAnswer
from langgraph_agentic_scientist.paper.references.plan import ReferencePlanAnswer
from langgraph_agentic_scientist.testing import FakeRetriever, structured_call


def record(key: str) -> ReferenceRecord:
    return ReferenceRecord(
        key=key,
        title=key,
        bibtex=f"@article{{{key},\n  title={{{key}}}\n}}",
        source="semantic_scholar",
        provenance={"id": f"ref.s2.{key}", "paper_id": key, "citation_key": key},
    )


def plan_answer(*queries: str) -> Any:
    return structured_call(
        ReferencePlanAnswer,
        {
            "memo": "Plan memo.",
            "plan": {
                "status": "ready",
                "central_topic": "Packing circles",
                "queries": [
                    {"query": q, "category": "core_method", "rationale": "grounded"}
                    for q in queries
                ],
            },
        },
    )


def gaps_ready() -> Any:
    return structured_call(
        DraftGapsAnswer,
        {
            "memo": "# Gaps\nOne.",
            "gaps": {
                "status": "ready",
                "summary": "Related work lacks comparators.",
                "missing_literatures": ["packing heuristics"],
                "rewrite_targets": [
                    {
                        "target_id": "rw-1",
                        "section": "Related Work",
                        "priority": "high",
                        "citation_intent": "comparator",
                        "reason": "uncited comparators",
                        "anchor_text": "prior heuristics",
                        "search_topics": ["packing heuristics"],
                    }
                ],
            },
        },
    )


async def test_default_references_and_postdraft_enrichment(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    retriever = FakeRetriever(
        {
            "circle packing": [record("A"), record("B")],
            "packing heuristics": [record("C"), record("A")],
        }
    )
    scripted = models(
        {
            Role.REFERENCE_PLANNER: [plan_answer("circle packing")],
            Role.PAPER_WRITER: [*write_paper(), *write_paper()],
            Role.DRAFT_GAP_ANALYST: [gaps_ready()],
            Role.POSTDRAFT_REFERENCE_PLANNER: [plan_answer("packing heuristics")],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted,
        retriever=retriever,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(min_refs=1, max_refs=5)),
    )
    graph = build_paper_graph(checkpointer=InMemorySaver())
    out = await run(graph, ctx, task)
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    bib = (await workspace.read_bytes(f"{paper_dir}/references.bib")).decode()
    for key in ("A", "C"):
        assert f"{{{key}," in bib
    for name in (
        "reference_query_plan.json",
        "draft_reference_gaps.json",
        "postdraft_reference_query_plan.json",
        "postdraft_references.bib",
        "references.pre_postdraft.bib",
        "postdraft_citation_weave_prompt.md",
        ".postdraft_reference_enrichment_complete",
    ):
        assert await workspace.exists(f"{paper_dir}/{name}"), name
    weave = scripted.scripted(Role.PAPER_WRITER).calls[3]
    text = next(m.content for m in weave if m.type == "human")
    assert "Status: `ready`" in text


async def test_no_gaps_skips_postdraft_retrieval(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.REFERENCE_PLANNER: [plan_answer("circle packing")],
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted,
        retriever=FakeRetriever({"circle packing": [record("A")]}),
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(min_refs=1, max_refs=5)),
    )
    out = await run(build_paper_graph(checkpointer=InMemorySaver()), ctx, task)
    assert out["paper"].status == "complete"
    assert not await workspace.exists(f"{out['paper'].paper_dir}/postdraft_references.bib")
