"""The harness-owned evaluation of an agent-authored candidate (upstream ``harness_eval.py``).

In a harness-evaluated attempt the agent only edits ``program_1``; the host then checks what the
agent left, evaluates the candidate once through the context's evaluator and writes the rubric
status itself. This module holds the parts of that step that are not a model call or an
evaluator call: the checks of what the agent left, the bounded reading of the candidate's
sources, the rubric status rules (:class:`HarnessRubricVerifier`), the files the host publishes
into the program directory, and the memory event of the evaluation.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.state import RubricResult, RubricSpec, RubricStatus, utc_now
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationResult
from ..prompts import PromptRegistry, render_text
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .context import FEEDBACK_KEY
from .ideas import rubric_json
from .rubrics import (
    COMPARATORS,
    candidate_source_tokens,
    missing_source_patterns,
    rubric_metric_check,
)

EVALUATOR_ARTIFACTS = (
    "metrics.json",
    "correct.json",
    "rubric_status.json",
    "stdout.log",
    "stderr.log",
    "evaluation_manifest.json",
    "harness_evaluate_stdout.log",
    "harness_evaluate_stderr.log",
)
"""Files an agent must not create in ``program_1`` (upstream ``EVALUATOR_ARTIFACTS``)."""

MAX_SOURCE_FILE_BYTES = 128 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_ENTRIES = 2048
MAX_SOURCE_DEPTH = 32
_LIMIT_ERROR = "program_1 source exceeds harness inspection limit"


class AgentOutputError(ValueError):
    """What the agent left does not satisfy the harness-evaluated attempt's contract."""


def feedback_strings(prompts: PromptRegistry) -> dict[str, Any]:
    """Parse the evidence and feedback strings of the registry (upstream's, as package data)."""
    return tomllib.loads(prompts.text(FEEDBACK_KEY))


async def ensure_agent_only_created_program_1(workspace: Workspace, experiment_dir: str) -> str:
    """Check what the candidate author left; return the entrypoint of ``program_1``.

    Upstream's rules: ``program_1`` is a real directory, no candidate directory other than
    ``program_0`` and ``program_1`` exists, no evaluator artifact exists in ``program_1``, and
    its ``program_spec.toml`` names an entrypoint that exists as a regular file.

    Raises:
        AgentOutputError: With upstream's message.
    """
    program_1 = join(experiment_dir, "program_1")
    entry = await workspace.stat(program_1)
    if entry is None or entry.kind != "dir":
        raise AgentOutputError("Harness-evaluated attempt requires agent-created program_1")
    for item in await workspace.list_tree(experiment_dir):
        top = item.path.split("/", 1)[0]
        is_dir = item.kind == "dir" or "/" in item.path
        if is_dir and top.startswith("program_") and top not in {"program_0", "program_1"}:
            raise AgentOutputError(
                f"Harness-evaluated attempt forbids extra candidate directories: {top}"
            )
    for artifact in EVALUATOR_ARTIFACTS:
        if await workspace.stat(join(program_1, artifact)) is not None:
            raise AgentOutputError(
                f"agent-created evaluator artifact is forbidden: {join(program_1, artifact)}"
            )
    try:
        entrypoint = await resolve_entrypoint(workspace, program_1)
    except Exception as exc:
        raise AgentOutputError(
            "program_1 must contain a valid program_spec.toml entrypoint"
        ) from exc
    target = await workspace.stat(join(program_1, entrypoint))
    if target is None or target.kind != "file":
        raise AgentOutputError(
            f"program_1 entrypoint does not exist: {join(program_1, entrypoint)}"
        )
    return entrypoint


async def read_candidate_sources(workspace: Workspace, program_dir: str) -> tuple[str, ...]:
    """Read every ``.py`` file of ``<program_dir>/src`` with upstream's fixed bounds.

    At most 2048 entries, depth 32, 128 KiB per file and 4 MiB in total; symlinks and
    non-regular files are refused.

    Raises:
        AgentOutputError: When a bound is exceeded or the tree is unsafe.
    """
    source_root = join(program_dir, "src")
    entry = await workspace.stat(source_root)
    if entry is None or entry.kind != "dir":
        raise AgentOutputError("program_1 source could not be safely inspected")
    sources: list[str] = []
    total = 0
    for count, item in enumerate(await workspace.list_tree(source_root), start=1):
        if count > MAX_SOURCE_ENTRIES or item.path.count("/") + 1 > MAX_SOURCE_DEPTH + 1:
            raise AgentOutputError(_LIMIT_ERROR)
        if item.kind == "dir":
            continue
        if item.kind == "symlink":
            raise AgentOutputError("program_1 source directory must not contain symlinks")
        if item.kind != "file":
            raise AgentOutputError("program_1 source must contain only regular files")
        total += item.size
        if item.size > MAX_SOURCE_FILE_BYTES or total > MAX_SOURCE_BYTES:
            raise AgentOutputError(_LIMIT_ERROR)
        if item.path.endswith(".py"):
            data = await workspace.read_bytes(join(source_root, item.path))
            sources.append(data.decode("utf-8", errors="replace"))
    return tuple(sources)


def exit_code_of(evaluation: EvaluationResult) -> int:
    """Return the exit code upstream would have seen for an evaluation's outcome."""
    if evaluation.status == "completed":
        return 0
    if evaluation.status == "timeout":
        return 124
    return evaluation.returncode if evaluation.returncode else 1


def _source_patterns(rubric: Mapping[str, Any]) -> list[str] | None:
    verification = rubric.get("verification")
    if not isinstance(verification, dict) or verification.get("type") != "source_contains":
        return None
    patterns = verification.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return None
    if not all(isinstance(pattern, str) and pattern for pattern in patterns):
        return None
    return patterns


def _metric_key_for_check(metric: str, metrics: Mapping[str, Any]) -> str:
    if metric in metrics:
        return metric
    if metric == "score" and "combined_score" in metrics:
        return "combined_score"
    return metric


def rubric_verdict(
    *,
    rubric: Mapping[str, Any],
    metrics: Mapping[str, Any],
    correct: bool,
    exit_code: int,
    missing_patterns: set[str],
    strings: Mapping[str, Any],
) -> tuple[bool, str]:
    """Return whether a rubric passed and the evidence (upstream ``_rubric_status``).

    A failed evaluation, or ``correct=false``, fails every rubric as "not assessed". A
    ``source_contains`` rubric passes iff every pattern occurs in the candidate's executable
    code; a metric rubric passes by its threshold; any other type fails.
    """
    spec = strings["rubric_evidence"]
    base = render_text(
        spec["base"],
        {
            "EXIT_CODE": str(exit_code),
            "CORRECT": str(correct),
            "COMBINED_SCORE": str(metrics.get("combined_score")),
        },
    )
    if exit_code != 0:
        return False, render_text(spec["evaluation_failed"], {"BASE": base})
    if not correct:
        return False, render_text(spec["acceptance_failed"], {"BASE": base})
    rubric_type = str(rubric.get("type", "")).strip().lower()
    patterns = _source_patterns(rubric)
    if patterns is not None:
        missing = [pattern for pattern in patterns if pattern in missing_patterns]
        if missing:
            return False, render_text(
                spec["missing_pattern"], {"BASE": base, "PATTERN": repr(missing[0])}
            )
        return True, render_text(
            spec["source_contains"], {"BASE": base, "COUNT": str(len(patterns))}
        )
    check = rubric_metric_check(dict(rubric))
    if check is None:
        reason = str(spec["no_threshold"])
        if rubric_type and rubric_type != "metric":
            reason = render_text(spec["cannot_verify"], {"RUBRIC_TYPE": rubric_type})
        return False, render_text(spec["no_threshold_evidence"], {"BASE": base, "REASON": reason})
    metric, comparator, threshold = check
    key = _metric_key_for_check(metric, metrics)
    raw = metrics.get(key)
    actual = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
    if actual is None:
        return False, render_text(spec["missing_metric"], {"BASE": base, "METRIC_KEY": key})
    passed = COMPARATORS[comparator](actual, threshold)
    return passed, render_text(
        spec["metric_compared"],
        {
            "BASE": base,
            "METRIC_KEY": key,
            "ACTUAL": f"{actual:g}",
            "COMPARATOR": comparator,
            "THRESHOLD": f"{threshold:g}",
        },
    )


class HarnessRubricVerifier:
    """Writes the rubric status of a harness-evaluated candidate (upstream's rules).

    This is the default :class:`~langgraph_agentic_scientist.core.interfaces.RubricVerifier` of the
    investigation phase. A caller replaces it through ``ScientistContext.rubric_verifier`` to
    verify rubrics some other way (for example with a judge model).

    Args:
        prompts: The registry holding the evidence strings (default: the package's).
    """

    def __init__(self, prompts: PromptRegistry | None = None) -> None:
        self._prompts = prompts or PromptRegistry.default()

    async def verify(
        self,
        *,
        rubrics: Sequence[RubricSpec],
        evaluation: EvaluationResult,
        program_dir: str,
        workspace: Workspace,
        attempt_id: str,
        idea_id: str,
    ) -> RubricStatus:
        """Return the rubric status of an evaluated program.

        Raises:
            AgentOutputError: When the candidate's sources cannot be inspected within bounds.
        """
        strings = feedback_strings(self._prompts)
        code = exit_code_of(evaluation)
        metrics = dict(evaluation.metrics)
        correct = bool(evaluation.correct)
        items = [rubric_json(rubric) for rubric in rubrics]
        sources = await read_candidate_sources(workspace, program_dir)
        tokens = candidate_source_tokens(sources)
        patterns = [pattern for item in items for pattern in (_source_patterns(item) or [])]
        missing = set(missing_source_patterns(tokens, patterns))
        results: list[RubricResult] = []
        for item in items:
            passed, evidence = rubric_verdict(
                rubric=item,
                metrics=metrics,
                correct=correct,
                exit_code=code,
                missing_patterns=missing,
                strings=strings,
            )
            assessment = (
                "not_assessed" if code != 0 or not correct else "passed" if passed else "failed"
            )
            results.append(
                RubricResult.model_validate(
                    {
                        "id": item["id"],
                        "pass": passed,
                        "assessment": assessment,
                        "evidence": evidence,
                    }
                )
            )
        return RubricStatus(
            attempt_id=attempt_id,
            idea_id=idea_id,
            overall_pass=bool(results) and all(result.passed for result in results),
            rubrics=results,
            metrics={},
            next_recommendation="generate a new idea",
        )


def status_json(status: RubricStatus) -> dict[str, Any]:
    """Return a rubric status as upstream writes ``rubric_status.json``."""
    return status.model_dump(mode="json", by_alias=True, exclude_none=True)


def evaluation_feedback(metrics: Mapping[str, Any], code: int, strings: Mapping[str, Any]) -> str:
    """Return the feedback line of an evaluation (upstream ``evaluation_feedback``)."""
    spec = strings["evaluation_feedback"]
    if code == 124:
        outcome = str(spec["timeout"])
    elif code != 0:
        outcome = render_text(spec["failed"], {"EXIT_CODE": str(code)})
    elif metrics.get("correct") is not True:
        outcome = str(spec["acceptance_failed"])
    else:
        outcome = str(spec["passed"])
    return render_text(
        spec["line"],
        {
            "OUTCOME": outcome,
            "METRICS": json.dumps(dict(metrics), sort_keys=True, ensure_ascii=True),
        },
    )


def memory_event(
    *,
    attempt_id: str,
    idea_id: str,
    metrics: Mapping[str, Any],
    correct: bool,
    feedback: str,
    strings: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return the ``evaluate`` memory event of a harness evaluation and its Markdown entry."""
    spec = strings["memory_event"]
    timestamp = utc_now()
    score = metrics.get("combined_score")
    evidence = render_text(
        spec["evidence"], {"SCORE": str(score), "CORRECT": str(correct), "FEEDBACK": feedback}
    )
    event = {
        "timestamp": timestamp,
        "step": spec["step"],
        "scope": spec["scope"],
        "summary": spec["summary"],
        "evidence": evidence,
        "next_action": spec["next_action"],
        "idea_id": idea_id,
        "attempt_id": attempt_id,
        "program_id": spec["program_id"],
        "metrics": dict(metrics),
    }
    markdown = (
        f"### {timestamp} | step={spec['step']} | scope={spec['scope']}\n"
        f"- summary: {spec['summary']}\n"
        f"- evidence: {evidence}\n"
        f"- next_action: {spec['next_action']}\n"
        f"- idea_id: {idea_id}\n"
        f"- attempt_id: {attempt_id}\n"
        f"- program_id: {spec['program_id']}\n\n"
    )
    return event, markdown


async def write_evaluation_artifacts(
    workspace: Workspace,
    program_dir: str,
    evaluation: EvaluationResult,
    *,
    attempt_id: str,
    idea_id: str,
) -> None:
    """Publish an evaluation's summaries into the program directory (upstream's layout).

    A completed evaluation publishes ``metrics.json`` and ``correct.json``; every evaluation
    publishes ``evaluation_manifest.json`` (producer ``harness_eval``, with the exit code) and the
    bounded diagnostic as ``harness_evaluate_stderr.log``. Evaluator-owned outputs stay with the
    evaluator.
    """
    code = exit_code_of(evaluation)
    if evaluation.status == "completed":
        await workspace.write_bytes(
            join(program_dir, "metrics.json"),
            (json.dumps(evaluation.metrics, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        await workspace.write_bytes(
            join(program_dir, "correct.json"),
            (
                json.dumps({"correct": bool(evaluation.correct)}, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
    now = utc_now()
    manifest: dict[str, Any] = {
        "producer": "harness_eval",
        "attempt_id": attempt_id,
        "idea_id": idea_id,
        "program_id": "program_1",
        "command": ["evaluator", evaluation.evaluation_id],
        "evaluation_id": evaluation.evaluation_id,
        "started_at": now,
        "finished_at": now,
        "exit_code": code,
        "returncode": code,
    }
    if evaluation.failure_category:
        manifest["failure_category"] = evaluation.failure_category
    await workspace.write_bytes(
        join(program_dir, "evaluation_manifest.json"),
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    await workspace.write_bytes(join(program_dir, "harness_evaluate_stdout.log"), b"")
    await workspace.write_bytes(
        join(program_dir, "harness_evaluate_stderr.log"), evaluation.summary.encode("utf-8")
    )


__all__ = [
    "EVALUATOR_ARTIFACTS",
    "AgentOutputError",
    "HarnessRubricVerifier",
    "ensure_agent_only_created_program_1",
    "evaluation_feedback",
    "exit_code_of",
    "feedback_strings",
    "memory_event",
    "read_candidate_sources",
    "rubric_verdict",
    "status_json",
    "write_evaluation_artifacts",
]
