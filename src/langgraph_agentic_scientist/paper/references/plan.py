"""Reference query plans: the structured answer of the reference planner and its validation.

Upstream ``reference_query_plan.py``. The planner (role R27, and the post-draft planner R32)
answers through structured output with a memo and a plan (adaptation A3: the memo-plus-JSON roles
return ``{"memo": <markdown>, "plan": <object>}``). The plan model is deliberately lenient about
values (every field has a default) so that the structured-output layer does not reject a plan
the method would repair: the semantic rules of upstream ``validate_reference_query_plan_payload``
run on the host (:func:`validate_reference_query_plan_payload`), and a failing plan is repaired
in a further round with feedback, as upstream does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from ...core.state import StateModel, utc_now
from ...workspace.paths import join
from ...workspace.protocol import Workspace

VALID_QUERY_CATEGORIES = frozenset(
    {
        "core_method",
        "baseline_comparator",
        "benchmark_context",
        "architectural_shift",
        "result_driven_theme",
    }
)
"""The categories a query of a plan may carry."""

MAX_REFERENCE_QUERY_COUNT = 10
"""The most queries a plan may hold (upstream ``MAX_REFERENCE_QUERY_COUNT``)."""

_LIST_FIELDS = (
    "core_methods",
    "comparators",
    "benchmarks",
    "architectural_shifts",
    "result_driven_themes",
    "forbidden_query_patterns",
)


class PlanQuery(StateModel):
    """One query of a reference query plan."""

    query: str = ""
    category: str = ""
    rationale: str = ""


class ReferencePlan(StateModel):
    """The reference query plan (``reference_query_plan.json``).

    ``status`` is ``ready`` with queries, or ``fail`` with a ``failure_reason`` when the grounded
    artifacts support no coherent literature neighbourhood.
    """

    status: str = "ready"
    central_topic: str = ""
    core_methods: list[str] = Field(default_factory=list)
    comparators: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    architectural_shifts: list[str] = Field(default_factory=list)
    result_driven_themes: list[str] = Field(default_factory=list)
    queries: list[PlanQuery] = Field(default_factory=list)
    forbidden_query_patterns: list[str] = Field(default_factory=list)
    failure_reason: str = ""


class ReferencePlanAnswer(StateModel):
    """The structured answer of a reference planner: a readable memo and the plan object."""

    memo: str
    plan: ReferencePlan


ReferenceQueryPlan = ReferencePlan
"""Alias: upstream calls the object a reference query plan."""


def validate_reference_query_plan_payload(
    payload: Mapping[str, Any],
    *,
    max_queries: int = MAX_REFERENCE_QUERY_COUNT,
) -> None:
    """Validate a reference query plan payload.

    Raises:
        ValueError: With upstream's message for the first rule the payload breaks.
    """
    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "fail"}:
        raise ValueError("reference_query_plan.json must set status to 'ready' or 'fail'")

    if status == "fail":
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if not failure_reason:
            raise ValueError(
                "reference_query_plan.json status=fail requires non-empty failure_reason"
            )
        return

    if not str(payload.get("central_topic", "")).strip():
        raise ValueError("reference_query_plan.json missing non-empty central_topic")

    query_rows = payload.get("queries")
    if not isinstance(query_rows, list) or not query_rows:
        raise ValueError("reference_query_plan.json must contain non-empty queries list")
    if len(query_rows) > max_queries:
        raise ValueError(
            f"reference_query_plan.json queries exceeds max {max_queries}: {len(query_rows)}"
        )

    seen_queries: set[str] = set()
    for idx, row in enumerate(query_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"reference_query_plan.json queries[{idx}] must be an object")
        query = " ".join(str(row.get("query", "")).split()).strip()
        rationale = " ".join(str(row.get("rationale", "")).split()).strip()
        category = str(row.get("category", "")).strip()
        if not query:
            raise ValueError(f"reference_query_plan.json queries[{idx}] missing non-empty query")
        if len(query) > 200:
            raise ValueError(f"reference_query_plan.json queries[{idx}].query exceeds 200 chars")
        if not rationale:
            raise ValueError(
                f"reference_query_plan.json queries[{idx}] missing non-empty rationale"
            )
        if category not in VALID_QUERY_CATEGORIES:
            allowed = ", ".join(sorted(VALID_QUERY_CATEGORIES))
            raise ValueError(
                f"reference_query_plan.json queries[{idx}].category must be one of: {allowed}"
            )
        key = query.lower()
        if key in seen_queries:
            raise ValueError(f"reference_query_plan.json queries[{idx}] duplicates query: {query}")
        seen_queries.add(key)

    for list_field in _LIST_FIELDS:
        value = payload.get(list_field, [])
        if not isinstance(value, list):
            raise ValueError(f"reference_query_plan.json field must be a list: {list_field}")


def reference_query_texts(payload: Mapping[str, Any]) -> list[str]:
    """Extract ordered query texts from a validated plan payload."""
    query_rows = payload.get("queries", [])
    out: list[str] = []
    for row in query_rows:
        if not isinstance(row, Mapping):
            continue
        query = " ".join(str(row.get("query", "")).split()).strip()
        if query:
            out.append(query)
    return out


def reference_query_plan_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact metadata from a validated query plan for provenance files."""
    return {
        "status": str(payload.get("status", "")).strip(),
        "central_topic": str(payload.get("central_topic", "")).strip(),
        "core_methods": payload.get("core_methods", []),
        "comparators": payload.get("comparators", []),
        "benchmarks": payload.get("benchmarks", []),
        "architectural_shifts": payload.get("architectural_shifts", []),
        "result_driven_themes": payload.get("result_driven_themes", []),
        "forbidden_query_patterns": payload.get("forbidden_query_patterns", []),
        "queries": payload.get("queries", []),
    }


async def write_plan_files(
    workspace: Workspace,
    paper_dir: str,
    *,
    stem: str = "reference_query_plan",
    memo: str,
    plan: Mapping[str, Any],
) -> None:
    """Write ``<stem>.md`` (the memo) and ``<stem>.json`` (the plan) into the paper directory.

    The harness writes both files from the structured answer, under the names the prompts use
    (``reference_query_plan`` for the plan, ``postdraft_reference_query_plan`` for the post-draft
    plan).
    """
    await workspace.write_bytes(join(paper_dir, f"{stem}.md"), memo.encode("utf-8"))
    await workspace.write_bytes(
        join(paper_dir, f"{stem}.json"),
        json.dumps(dict(plan), indent=2, sort_keys=True).encode("utf-8"),
    )


async def write_reference_query_plan_validation_feedback(
    workspace: Workspace,
    paper_dir: str,
    *,
    stem: str = "reference_query_plan",
    max_queries: int = MAX_REFERENCE_QUERY_COUNT,
) -> dict[str, Any]:
    """Validate the plan artifacts of ``stem`` and persist the feedback (upstream function).

    Writes ``<stem>_validation_feedback.json`` and ``.md`` and returns the feedback mapping
    (``ok`` and ``hard_errors``).
    """
    plan_md = join(paper_dir, f"{stem}.md")
    plan_json = join(paper_dir, f"{stem}.json")
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": workspace.host_path(paper_dir),
        "ok": False,
        "hard_errors": [],
    }
    try:
        if not (entry := await workspace.stat(plan_md)) or entry.kind != "file":
            raise ValueError(
                f"Missing required reference query plan memo: {workspace.host_path(plan_md)}"
            )
        if not (entry := await workspace.stat(plan_json)) or entry.kind != "file":
            raise ValueError(
                f"Missing required reference query plan JSON: {workspace.host_path(plan_json)}"
            )
        if not (await workspace.read_bytes(plan_md)).decode("utf-8", errors="replace").strip():
            raise ValueError(f"Reference query plan memo is empty: {workspace.host_path(plan_md)}")
        payload = json.loads((await workspace.read_bytes(plan_json)).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {workspace.host_path(plan_json)}")
        validate_reference_query_plan_payload(payload, max_queries=max_queries)
        if str(payload.get("status", "")).strip() == "fail":
            raise ValueError(
                f"reference query planning failed: {str(payload.get('failure_reason', '')).strip()}"
            )
    except (ValueError, UnicodeDecodeError) as exc:
        feedback["hard_errors"] = [str(exc)]
    else:
        feedback["ok"] = True

    await workspace.write_bytes(
        join(paper_dir, f"{stem}_validation_feedback.json"),
        json.dumps(feedback, indent=2, sort_keys=True).encode("utf-8"),
    )
    lines = [
        "# Reference query plan validation feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    hard_errors = feedback.get("hard_errors", [])
    if isinstance(hard_errors, list) and hard_errors:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in hard_errors)
        lines.append("")
    await workspace.write_bytes(
        join(paper_dir, f"{stem}_validation_feedback.md"),
        (("\n".join(lines).strip()) + "\n").encode("utf-8"),
    )
    return feedback


__all__ = [
    "MAX_REFERENCE_QUERY_COUNT",
    "VALID_QUERY_CATEGORIES",
    "PlanQuery",
    "ReferencePlan",
    "ReferencePlanAnswer",
    "ReferenceQueryPlan",
    "reference_query_plan_metadata",
    "reference_query_texts",
    "validate_reference_query_plan_payload",
    "write_plan_files",
    "write_reference_query_plan_validation_feedback",
]
