"""Builders shared by the task design tests: designs, reviews, patches and scripted answers.

Not a test module: it holds the payload builders (modelled on the fixtures of upstream's
``tests/task_drafting/test_task_design.py``) and the scripted answers of the roles. The tests
import it as ``test_design_support``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph_agentic_scientist.drafting.design_models import (
    DESIGN_DIMENSIONS,
    DeltaAudit,
    DesignPatch,
    DesignReview,
    TaskDesign,
)
from langgraph_agentic_scientist.testing import structured_call

TASK_MD = "# Goal\nTrain the requested method.\n\n# Rubrics\n- Improve held-out loss.\n"
BUDGET = "/planes/search/budget"
CLAIM = "/method/claim"
FINGERPRINT_RE = re.compile(r"Expected design fingerprint: `([0-9a-f]{64})`")
CANDIDATE_FINGERPRINT_RE = re.compile(r"Expected candidate fingerprint: `([0-9a-f]{64})`")


def task_sha256(task_md: str = TASK_MD) -> str:
    """Return the SHA-256 of the task text."""
    return hashlib.sha256(task_md.encode("utf-8")).hexdigest()


def design_payload(task_md: str = TASK_MD) -> dict[str, Any]:
    """Return a valid TaskDesign v1 object for ``task_md``."""
    budget = {
        "unit": "outer updates",
        "training_steps": 160,
        "wall_clock_seconds": 3600,
        "hardware": "4x GPU",
        "precision": "bfloat16",
    }
    plane = {
        "resources": ["pinned upstream benchmark"],
        "seeds": [1],
        "budget": budget,
        "baselines": ["published baseline"],
        "metrics": ["held-out loss"],
        "correctness": ["finite metrics", "completed training"],
        "smoke": "one update through the same candidate and metric path",
    }
    return {
        "schema_version": 1,
        "canonical_task": {"path": "task.md", "sha256": task_sha256(task_md)},
        "method": {"class": "learned optimization", "claim": "Improve held-out loss."},
        "upstream": [
            {
                "repository": "https://example.test/upstream.git",
                "revision": "0123456789abcdef",
                "license": "Apache-2.0",
                "entrypoints": ["train.py"],
            }
        ],
        "candidate": {
            "interface": "python_rpc_v1:handle",
            "editable_paths": ["program_0/src"],
            "state_semantics": "Reset all candidate state for every declared seed.",
        },
        "planes": {
            "search": plane,
            "final": {**deepcopy(plane), "seeds": [2], "resources": ["sealed held-out benchmark"]},
        },
        "feedback": {
            "search_visible": ["search metrics"],
            "final_sealed": ["final identities", "final metrics"],
        },
        "publication": {"required_evidence": ["sealed final evaluation"]},
    }


def make_design(mutate: Callable[[dict[str, Any]], None] | None = None) -> TaskDesign:
    """Return a design object, optionally mutated before it is built (``mutate`` edits a dict)."""
    payload = design_payload()
    if mutate is not None:
        mutate(payload)
    return TaskDesign.model_validate(payload)


def review_payload(
    fingerprint: str,
    failures: Mapping[str, str] | None = None,
    *,
    design_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a schema-3 review that fails the given dimensions.

    Args:
        fingerprint: The reviewed design's fingerprint.
        failures: Failed dimension id to the JSON pointer it may repair.
        design_paths: Optional evidence pointer per dimension (default: the repair path, or the
            budget).
    """
    failures = failures or {}
    design_paths = design_paths or {}
    dimensions = []
    for identifier in DESIGN_DIMENSIONS:
        failed = identifier in failures
        path = failures.get(identifier, BUDGET)
        evidence_path = design_paths.get(identifier, path)
        dimensions.append(
            {
                "id": identifier,
                "verdict": "fail" if failed else "pass",
                "evidence": [f"Concrete evidence for {identifier}."],
                "design_paths": list(dict.fromkeys([evidence_path, path])),
                "task_evidence": ["task.md:1: Goal."],
                "repair_paths": [path] if failed else [],
                "remediation": f"Repair {identifier}." if failed else "",
            }
        )
    return {
        "schema_version": 3,
        "verdict": "fail" if failures else "pass",
        "summary": "Reviewed design.",
        "design_fingerprint": fingerprint,
        "dimensions": dimensions,
    }


def delta_payload(
    parent_fingerprint: str,
    candidate_fingerprint: str,
    *,
    fixed: Sequence[str],
    regressions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a schema-2 delta audit."""
    return {
        "schema_version": 2,
        "verdict": "fail" if regressions else "pass",
        "summary": "Delta audit.",
        "parent_fingerprint": parent_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "fixed_dimensions": list(fixed),
        "regressions": [dict(row) for row in regressions],
    }


def regression(
    dimension: str, path: str, repair_paths: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return one regression row of a delta audit."""
    return {
        "dimension": dimension,
        "design_paths": [path],
        "repair_paths": list(repair_paths) if repair_paths is not None else [path],
        "evidence": [f"The candidate regressed {dimension}."],
        "remediation": f"Repair {dimension}.",
    }


def patch_payload(
    parent_fingerprint: str, path: str, value: Any, *, op: str = "replace"
) -> dict[str, Any]:
    """Return a one-operation patch."""
    return {
        "schema_version": 1,
        "parent_fingerprint": parent_fingerprint,
        "operations": [{"op": op, "path": path, "value": value}],
    }


# --- scripted answers ---------------------------------------------------------------------------


def prompt_of(messages: Sequence[BaseMessage]) -> str:
    """Return the text of the first human message of a session (the rendered prompt)."""
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    raise AssertionError("no human message in the session")


def fingerprint_in(messages: Sequence[BaseMessage]) -> str:
    """Return the fingerprint a judge prompt names."""
    match = FINGERPRINT_RE.search(prompt_of(messages))
    assert match, "the prompt names no expected design fingerprint"
    return match.group(1)


def judge(
    failures: Mapping[str, str] | None = None,
    *,
    seen: list[str] | None = None,
    design_paths: Mapping[str, str] | None = None,
) -> Callable[[list[BaseMessage]], AIMessage]:
    """A judge answer that reviews the fingerprint the prompt names (and records the prompt)."""

    def answer(messages: list[BaseMessage]) -> AIMessage:
        if seen is not None:
            seen.append(prompt_of(messages))
        return structured_call(
            DesignReview,
            review_payload(fingerprint_in(messages), failures, design_paths=design_paths),
        )

    return answer


def patcher(
    path: str,
    value: Any,
    *,
    seen: list[str] | None = None,
    op: str = "replace",
    parent_fingerprint: str | None = None,
) -> Callable[[list[BaseMessage]], AIMessage]:
    """A patch answer that replaces ``path`` (the parent fingerprint is read from the prompt)."""

    def answer(messages: list[BaseMessage]) -> AIMessage:
        text = prompt_of(messages)
        if seen is not None:
            seen.append(text)
        match = re.search(r"`parent_fingerprint` equal to\s+`([0-9a-f]{64})`", text)
        assert match, "the repair prompt names no parent fingerprint"
        return structured_call(
            DesignPatch,
            patch_payload(parent_fingerprint or match.group(1), path, value, op=op),
        )

    return answer


def auditor(
    *,
    fixed: Sequence[str],
    regressions: Sequence[Mapping[str, Any]] = (),
    seen: list[str] | None = None,
    parent_fingerprint: str | None = None,
) -> Callable[[list[BaseMessage]], AIMessage]:
    """A delta audit answer for the candidate the prompt names."""

    def answer(messages: list[BaseMessage]) -> AIMessage:
        text = prompt_of(messages)
        if seen is not None:
            seen.append(text)
        candidate = CANDIDATE_FINGERPRINT_RE.search(text)
        assert candidate, "the audit prompt names no candidate fingerprint"
        return structured_call(
            DeltaAudit,
            delta_payload(
                parent_fingerprint or "0" * 64,
                candidate.group(1),
                fixed=fixed,
                regressions=regressions,
            ),
        )

    return answer
