"""Semantic Scholar retrieval (upstream ``semantic_scholar.py``) as a ``ReferenceRetriever``.

:class:`SemanticScholarRetriever` answers one query at a time through the Graph API: it searches,
ranks the candidates by citation count, and fetches each candidate's details (with its BibTeX)
until it has as many usable references as were asked for. The references graph calls it once per
query and stops when it has enough. Retrieval beyond Semantic Scholar (the vendored OpenAlex and
``scholarly`` scripts of the ``retrieve-references`` skill, which upstream runs when Semantic
Scholar fails or falls short) is :meth:`SemanticScholarRetriever.fallback`.

Differences from upstream, recorded in ``docs/adaptations/paper_host.md``: the package keeps no
retry loop and no sleeps of its own. A rate-limited or forbidden response raises
:class:`SemanticScholarRateLimited`, which the references graph retries with its node
``RetryPolicy`` (upstream retried each request up to five times, waiting the ``Retry-After``
time); a failure of one candidate's details skips that candidate, as upstream does.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ...core.interfaces import ReferenceQuery, ReferenceRecord
from ...core.state import utc_now
from ...workspace.protocol import Workspace
from .bibtex import CITATION_KEY_RE, extract_bibtex_key
from .retrieve import FallbackOutcome, run_script_fallbacks

SEMANTIC_SCHOLAR_GRAPH_API = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarError(RuntimeError):
    """Retrieval from Semantic Scholar failed."""


class SemanticScholarRateLimited(SemanticScholarError):
    """Semantic Scholar answered ``403`` or ``429``; the caller retries after a pause."""


def resolve_s2_api_key(explicit_key: str | None = None) -> str | None:
    """Return the API key (the argument, else ``S2_API_KEY`` of the environment), or ``None``."""
    if explicit_key:
        return explicit_key.strip() or None
    env_key = os.environ.get("S2_API_KEY")
    if env_key is None:
        return None
    env_key = env_key.strip()
    return env_key or None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def build_search_url(query: str, *, limit: int) -> str:
    """Return the Graph API search URL of a query."""
    params = urllib.parse.urlencode(
        {"query": query, "limit": str(limit), "fields": "paperId,title,year,venue,citationCount"}
    )
    return f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/search?{params}"


def build_paper_url(paper_id: str) -> str:
    """Return the Graph API URL of one paper's details (including its BibTeX)."""
    safe_paper_id = urllib.parse.quote(paper_id, safe="")
    params = urllib.parse.urlencode(
        {"fields": "paperId,title,year,venue,citationCount,citationStyles,url"}
    )
    return f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/{safe_paper_id}?{params}"


class SemanticScholarRetriever:
    """The default :class:`~langgraph_agentic_scientist.core.interfaces.ReferenceRetriever`.

    Args:
        api_key: The Semantic Scholar API key (default: ``S2_API_KEY`` of the environment;
            requests are anonymous without one).
        timeout_seconds: The timeout of each request (upstream: 20 s).
        client: An ``httpx.AsyncClient`` to use (a test passes one with a mock transport); a
            client is created per call otherwise.
        python: The interpreter that runs the fallback scripts through the workspace (default:
            the current interpreter; pass ``"python3"`` for a workspace that runs elsewhere).
        prompts: Where the fallback scripts come from (default: the package's skills).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
        python: str | None = None,
        prompts: Any = None,
    ) -> None:
        self.api_key = resolve_s2_api_key(api_key)
        self.timeout_seconds = timeout_seconds
        self.client = client
        self.python = python
        self.prompts = prompts

    # --- HTTP ----------------------------------------------------------------------------

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            response = await client.get(url, headers=headers, timeout=self.timeout_seconds)
        except httpx.HTTPError as exc:
            raise SemanticScholarError(f"Semantic Scholar request failed for {url}: {exc}") from exc
        if response.status_code in {403, 429}:
            raise SemanticScholarRateLimited(
                f"Semantic Scholar HTTP {response.status_code} for {url}"
            )
        if response.status_code >= 400:
            raise SemanticScholarError(f"Semantic Scholar HTTP {response.status_code} for {url}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SemanticScholarError(
                f"Invalid JSON from Semantic Scholar for {url}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticScholarError(f"Unexpected Semantic Scholar response type for {url}")
        return payload

    # --- ReferenceRetriever ----------------------------------------------------------------

    async def search(self, query: ReferenceQuery, *, workspace: Workspace) -> list[ReferenceRecord]:
        """Return up to ``query.limit`` usable references for one query, best cited first.

        ``query.filters`` may carry ``search_limit`` (candidates per search, default 25),
        ``exclude_paper_ids`` and ``exclude_keys`` (references already accepted for earlier
        queries). A candidate whose details cannot be fetched, or that has no BibTeX or a
        citation key already taken, is skipped.

        Raises:
            SemanticScholarRateLimited: On ``403`` or ``429`` (the caller retries).
            SemanticScholarError: When the search itself fails.
        """
        if self.client is not None:
            return await self._search(self.client, query)
        async with httpx.AsyncClient() as client:
            return await self._search(client, query)

    async def _search(
        self, client: httpx.AsyncClient, query: ReferenceQuery
    ) -> list[ReferenceRecord]:
        filters: Mapping[str, Any] = query.filters
        search_limit = int(filters.get("search_limit", 25))
        excluded_ids = {str(item) for item in filters.get("exclude_paper_ids", ())}
        used_keys = {str(item) for item in filters.get("exclude_keys", ())}
        text = " ".join(query.query.split()).strip()
        payload = await self._get_json(client, build_search_url(text, limit=search_limit))
        data = payload.get("data")
        rows = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set(excluded_ids)
        for row in rows:
            paper_id = row.get("paperId")
            if not isinstance(paper_id, str) or not paper_id.strip():
                continue
            paper_id = paper_id.strip()
            if paper_id in seen:
                continue
            seen.add(paper_id)
            candidates.append(
                {
                    "paper_id": paper_id,
                    "citation_count": _as_int(row.get("citationCount")) or 0,
                }
            )
        ranked = sorted(
            candidates, key=lambda item: (-int(item["citation_count"]), item["paper_id"])
        )

        accepted: list[ReferenceRecord] = []
        for candidate in ranked:
            if len(accepted) >= query.limit:
                break
            paper_id = str(candidate["paper_id"])
            try:
                details = await self._get_json(client, build_paper_url(paper_id))
            except SemanticScholarRateLimited:
                raise
            except SemanticScholarError:
                continue
            record = self._record(details, paper_id, text, int(candidate["citation_count"]))
            if record is None or record.key in used_keys:
                continue
            used_keys.add(record.key)
            accepted.append(record)
        return accepted

    @staticmethod
    def _record(
        details: Mapping[str, Any], paper_id: str, query: str, search_citation_count: int
    ) -> ReferenceRecord | None:
        citation_styles = details.get("citationStyles")
        bibtex = None
        if isinstance(citation_styles, dict):
            maybe_bibtex = citation_styles.get("bibtex")
            if isinstance(maybe_bibtex, str) and maybe_bibtex.strip():
                bibtex = maybe_bibtex.strip()
        if bibtex is None:
            return None
        citation_key = extract_bibtex_key(bibtex)
        if citation_key is None:
            return None
        detail_citation_count = _as_int(details.get("citationCount"))
        citation_count = (
            detail_citation_count if detail_citation_count is not None else search_citation_count
        )
        ref_id = f"ref.s2.{paper_id}"
        retrieved_at = utc_now()
        title = details.get("title")
        venue = details.get("venue")
        year = _as_int(details.get("year"))
        provenance = {
            "id": ref_id,
            "paper_id": paper_id,
            "citation_key": citation_key,
            "title": title,
            "year": year,
            "venue": venue,
            "citation_count": citation_count,
            "query": query,
            "source_url": build_paper_url(paper_id),
            "retrieved_at": retrieved_at,
        }
        return ReferenceRecord(
            key=citation_key,
            title=title if isinstance(title, str) else "",
            year=year,
            venue=venue if isinstance(venue, str) else None,
            bibtex=bibtex,
            source="semantic_scholar",
            provenance=provenance,
        )

    # --- fallbacks -------------------------------------------------------------------------

    async def fallback(
        self,
        queries: Sequence[str],
        *,
        min_refs: int,
        max_refs: int,
        paper_dir: str,
        workspace: Workspace,
        s2_error: str | None,
        s2_count: int,
    ) -> FallbackOutcome:
        """Run the OpenAlex and ``scholarly`` fallbacks of upstream through the workspace.

        Used when Semantic Scholar failed or returned fewer than ``min_refs`` references. The
        vendored scripts of the ``retrieve-references`` skill are written into the paper
        directory and run there; see :func:`~.retrieve.run_script_fallbacks`.
        """
        return await run_script_fallbacks(
            workspace,
            paper_dir,
            queries=queries,
            min_refs=min_refs,
            max_refs=max_refs,
            s2_error=s2_error,
            s2_count=s2_count,
            python=self.python,
            prompts=self.prompts,
        )


__all__ = [
    "CITATION_KEY_RE",
    "SEMANTIC_SCHOLAR_GRAPH_API",
    "SemanticScholarError",
    "SemanticScholarRateLimited",
    "SemanticScholarRetriever",
    "build_paper_url",
    "build_search_url",
    "extract_bibtex_key",
    "resolve_s2_api_key",
]
