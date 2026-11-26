"""Semantic Scholar retrieval (against a mock transport) and the script fallbacks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.core.interfaces import ReferenceQuery
from langgraph_agentic_scientist.paper.references import (
    SemanticScholarError,
    SemanticScholarRateLimited,
    SemanticScholarRetriever,
    reference_provenance_payload,
    retrieval_failure_message,
    run_script_fallbacks,
    write_reference_outputs,
)
from langgraph_agentic_scientist.paper.references.retrieve import SCRIPTS_DIR, row_to_record
from langgraph_agentic_scientist.paper.references.semantic_scholar import (
    build_paper_url,
    resolve_s2_api_key,
)
from langgraph_agentic_scientist.workspace import CommandResult


def paper(title: str, year: int, citations: int, bibtex: str | None) -> dict[str, Any]:
    return {
        "title": title,
        "year": year,
        "venue": "V",
        "citationCount": citations,
        "bibtex": bibtex,
    }


PAPERS: dict[str, dict[str, Any]] = {
    "p-low": paper("Low", 2019, 3, "@article{Low2019,\n}"),
    "p-high": paper("High", 2020, 900, "@article{High2020,\n}"),
    "p-mid": paper("Mid", 2021, 50, "@article{Mid2021,\n}"),
    "p-nobib": paper("None", 2022, 700, None),
    "p-dup": paper("Dup", 2023, 40, "@article{High2020,\n}"),
}


def transport(
    *,
    search_status: int = 200,
    fail_details: set[str] | None = None,
    rate_limit: set[str] | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = urlparse(str(request.url)).path
        if path.endswith("/paper/search"):
            if search_status != 200:
                return httpx.Response(search_status)
            data = [
                {"paperId": pid, "citationCount": p["citationCount"]} for pid, p in PAPERS.items()
            ]
            data.append({"paperId": "", "citationCount": 1})
            return httpx.Response(200, json={"data": data})
        pid = path.rsplit("/", 1)[-1]
        if pid in (rate_limit or set()):
            return httpx.Response(429, headers={"Retry-After": "1"})
        if pid in (fail_details or set()):
            return httpx.Response(500)
        paper = PAPERS[pid]
        styles = {"bibtex": paper["bibtex"]} if paper["bibtex"] else None
        return httpx.Response(
            200,
            json={
                "paperId": pid,
                "title": paper["title"],
                "year": paper["year"],
                "venue": paper["venue"],
                "citationCount": paper["citationCount"] + 1,
                "citationStyles": styles,
            },
        )

    return httpx.MockTransport(handler), seen


async def search(
    workspace: LocalWorkspace,
    transport_: httpx.MockTransport,
    query: ReferenceQuery,
    **options: Any,
) -> list[Any]:
    async with httpx.AsyncClient(transport=transport_) as client:
        retriever = SemanticScholarRetriever(client=client, **options)
        return await retriever.search(query, workspace=workspace)


async def test_candidates_are_ranked_by_citations_and_cut_at_the_limit(
    workspace: LocalWorkspace,
) -> None:
    mock, seen = transport()
    records = await search(workspace, mock, ReferenceQuery(query="  circle  packing ", limit=2))
    assert [record.key for record in records] == ["High2020", "Mid2021"]
    # the search asked for 25 candidates of the normalised query; details were fetched in rank order
    first = parse_qs(urlparse(str(seen[0].url)).query)
    assert first["query"] == ["circle packing"] and first["limit"] == ["25"]
    fetched = [urlparse(str(request.url)).path.rsplit("/", 1)[-1] for request in seen[1:]]
    assert fetched == ["p-high", "p-nobib", "p-mid"]  # 'p-nobib' has no BibTeX and was skipped
    record = records[0]
    assert record.source == "semantic_scholar" and record.title == "High" and record.year == 2020
    assert record.provenance["id"] == "ref.s2.p-high" and record.provenance["citation_count"] == 901
    assert record.provenance["source_url"] == build_paper_url("p-high")
    assert record.provenance["query"] == "circle packing"


async def test_accepted_keys_and_papers_of_earlier_queries_are_excluded(
    workspace: LocalWorkspace,
) -> None:
    mock, seen = transport()
    query = ReferenceQuery(
        query="q",
        limit=5,
        filters={"search_limit": 7, "exclude_paper_ids": ["p-mid"], "exclude_keys": ["High2020"]},
    )
    records = await search(workspace, mock, query)
    assert [record.key for record in records] == ["Low2019"]  # High and Dup carry the taken key
    assert parse_qs(urlparse(str(seen[0].url)).query)["limit"] == ["7"]
    assert "p-mid" not in [urlparse(str(r.url)).path.rsplit("/", 1)[-1] for r in seen]


async def test_a_failed_candidate_is_skipped_but_a_rate_limit_is_raised(
    workspace: LocalWorkspace,
) -> None:
    mock, _ = transport(fail_details={"p-high"})
    records = await search(workspace, mock, ReferenceQuery(query="q", limit=1))
    assert [record.key for record in records] == ["Mid2021"]
    mock, _ = transport(rate_limit={"p-high"})
    with pytest.raises(SemanticScholarRateLimited, match="HTTP 429"):
        await search(workspace, mock, ReferenceQuery(query="q", limit=1))
    mock, _ = transport(search_status=403)
    with pytest.raises(SemanticScholarRateLimited, match="HTTP 403"):
        await search(workspace, mock, ReferenceQuery(query="q"))
    mock, _ = transport(search_status=500)
    with pytest.raises(SemanticScholarError, match="HTTP 500") as error:
        await search(workspace, mock, ReferenceQuery(query="q"))
    assert not isinstance(error.value, SemanticScholarRateLimited)


async def test_the_api_key_is_sent_when_given(
    workspace: LocalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock, seen = transport()
    await search(workspace, mock, ReferenceQuery(query="q", limit=1), api_key=" secret ")
    assert seen[0].headers["x-api-key"] == "secret"
    monkeypatch.setenv("S2_API_KEY", " from-env ")
    assert resolve_s2_api_key() == "from-env" and resolve_s2_api_key("x") == "x"
    monkeypatch.delenv("S2_API_KEY")
    assert resolve_s2_api_key() is None
    mock, seen = transport()
    await search(workspace, mock, ReferenceQuery(query="q", limit=1))
    assert "x-api-key" not in seen[0].headers


async def test_a_network_error_is_a_retrieval_error(workspace: LocalWorkspace) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(SemanticScholarError, match="request failed"):
        await search(workspace, httpx.MockTransport(boom), ReferenceQuery(query="q"))


# --- the script fallbacks ---


class ScriptWorkspace(LocalWorkspace):
    """A workspace whose fallback scripts are scripted: each call writes the given rows."""

    def __init__(self, root: Any, outcomes: dict[str, dict[str, Any]]) -> None:
        super().__init__(root)
        self.outcomes = outcomes
        self.calls: list[list[str]] = []

    async def run(self, argv: Sequence[str], **kwargs: Any) -> CommandResult:  # type: ignore[override]
        script = argv[1].rsplit("/", 1)[-1]
        if script not in self.outcomes:
            return await super().run(argv, **kwargs)
        self.calls.append(list(argv))
        outcome = self.outcomes[script]
        cwd = kwargs["cwd"]
        if outcome.get("keys"):
            stem = (
                "references.openalex" if script.startswith("openalex") else "references.scholarly"
            )
            bib = "\n\n".join(f"@article{{{key},\n  title={{{key}}}\n}}" for key in outcome["keys"])
            await self.write_bytes(f"{cwd}/{stem}.bib", bib.encode())
            rows = [
                {"id": f"{stem}.{key}", "citation_key": key, "title": key}
                for key in outcome["keys"]
            ]
            await self.write_bytes(f"{cwd}/{stem}.json", json.dumps({"references": rows}).encode())
        return CommandResult(
            argv=list(argv),
            returncode=outcome.get("returncode", 0),
            stdout=outcome.get("stdout", ""),
            stderr=outcome.get("stderr", ""),
            timed_out=outcome.get("timed_out", False),
        )


def option(call: list[str], name: str) -> str:
    return call[call.index(name) + 1]


async def test_the_openalex_script_is_the_first_fallback(tmp_path: Any) -> None:
    workspace = ScriptWorkspace(
        tmp_path / "ws", {"openalex_fetch.py": {"keys": ["A1", "A2", "A3"]}}
    )
    outcome = await run_script_fallbacks(
        workspace, "paper", queries=["q1", "q2"], min_refs=3, max_refs=5, s2_error=None, s2_count=1
    )
    assert [record.key for record in outcome.records] == ["A1", "A2", "A3"]
    assert {record.source for record in outcome.records} == {"openalex"}
    assert outcome.sources == ["openalex"] and outcome.openalex_error is None
    (call,) = workspace.calls
    assert call[0] and call[1].endswith(f"{SCRIPTS_DIR}/openalex_fetch.py")
    assert option(call, "--min-refs") == "3" and option(call, "--max-refs") == "5"
    assert option(call, "--search-limit") == "10" and "--allow-partial-output" in call
    assert [call[i + 1] for i, arg in enumerate(call) if arg == "--query"] == ["q1", "q2"]
    assert option(call, "--out-bib") == "references.openalex.bib"
    assert await workspace.exists(f"paper/{SCRIPTS_DIR}/scholarly_fetch.py")


async def test_scholarly_asks_for_the_missing_references_and_merges_after_openalex(
    tmp_path: Any,
) -> None:
    workspace = ScriptWorkspace(
        tmp_path / "ws",
        {
            "openalex_fetch.py": {"keys": ["A1"], "returncode": 1, "stderr": "boom\n" * 30},
            "scholarly_fetch.py": {"keys": ["A1", "S1", "S2"]},
        },
    )
    outcome = await run_script_fallbacks(
        workspace, "paper", queries=["q"], min_refs=3, max_refs=6, s2_error="s2 down", s2_count=0
    )
    assert outcome.openalex_error is not None and outcome.openalex_error.count("boom") == 20
    assert [(r.key, r.source) for r in outcome.records] == [
        ("A1", "openalex"), ("S1", "scholarly"), ("S2", "scholarly"),
    ]  # fmt: skip
    assert outcome.sources == ["openalex", "scholarly"]
    scholarly = workspace.calls[1]
    assert option(scholarly, "--min-refs") == "2" and option(scholarly, "--max-refs") == "5"
    assert option(scholarly, "--total-timeout-seconds") == "300"
    assert option(scholarly, "--progress-interval-seconds") == "10"


async def test_no_fallback_runs_when_semantic_scholar_is_enough_and_failures_are_reported(
    tmp_path: Any,
) -> None:
    workspace = ScriptWorkspace(
        tmp_path / "ws",
        {
            "openalex_fetch.py": {"timed_out": True},
            "scholarly_fetch.py": {"returncode": 2, "stdout": "partial out"},
        },
    )
    outcome = await run_script_fallbacks(
        workspace, "paper", queries=["q"], min_refs=3, max_refs=5, s2_error="s2 down", s2_count=0
    )
    assert outcome.records == []
    assert outcome.openalex_error == "OpenAlex fallback timed out after 240s"
    assert outcome.scholarly_error == "partial out"
    skipped = ScriptWorkspace(
        tmp_path / "ws2", {"openalex_fetch.py": {}, "scholarly_fetch.py": {"keys": ["S1"]}}
    )
    result = await run_script_fallbacks(
        skipped, "paper", queries=["q"], min_refs=1, max_refs=2, s2_error=None, s2_count=1
    )
    assert [call[1].rsplit("/", 1)[-1] for call in skipped.calls] == ["scholarly_fetch.py"]
    assert [r.key for r in result.records] == ["S1"]


async def test_the_failure_message_names_each_provider_that_failed() -> None:
    assert retrieval_failure_message("s2", "oa", "sc") == (
        "Reference retrieval failed. Semantic Scholar: s2; OpenAlex: oa; scholarly: sc"
    )
    assert retrieval_failure_message("s2", None, "sc") == (
        "Reference retrieval failed. Semantic Scholar: s2; scholarly: sc"
    )
    assert retrieval_failure_message(None, "oa", "sc") == (
        "Reference retrieval failed. OpenAlex: oa; scholarly: sc"
    )
    assert retrieval_failure_message(None, None, "sc") == (
        "Reference retrieval failed via scholarly fallback: sc"
    )
    assert retrieval_failure_message(None, "oa", None) == (
        "Reference retrieval failed via OpenAlex fallback: oa"
    )
    assert retrieval_failure_message("s2", None, None) == "s2"
    assert retrieval_failure_message(None, None, None) == (
        "Reference retrieval failed without producing references.bib"
    )


async def test_reference_files_are_written_with_upstreams_payload(
    workspace: LocalWorkspace,
) -> None:
    rows = [
        {"id": "ref.s2.a", "citation_key": "A", "title": "T", "year": 2020, "bibtex": "@a{A,}"},
        {"id": "ref.s2.b", "citation_key": "B", "title": "U", "bibtex": "@b{B,}"},
    ]
    records = [row_to_record(row, source="semantic_scholar") for row in rows]
    assert records[0].year == 2020 and records[1].year is None
    payload = reference_provenance_payload(
        records,
        generated_at="t",
        results_dir="r",
        campaign_results_dir="c",
        queries=["q"],
        plan_metadata={"status": "ready"},
    )
    assert payload["count"] == 2 and payload["references"][0] == {
        "id": "ref.s2.a", "citation_key": "A", "title": "T", "year": 2020,
    }  # fmt: skip
    assert set(payload) == {
        "generated_at", "results_dir", "campaign_results_dir", "queries",
        "reference_query_plan", "count", "references",
    }  # fmt: skip
    await write_reference_outputs(workspace, "paper", records, payload)
    assert (await workspace.read_bytes("paper/references.bib")).decode() == "@a{A,}\n\n@b{B,}\n"
    stored = json.loads((await workspace.read_bytes("paper/references_provenance.json")).decode())
    assert stored == payload
