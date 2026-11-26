"""The AI review graph (upstream ``src/aireview.py`` and the pipeline's final review phase).

The method, as upstream defines it: extract the manuscript text from the PDF; ask five reviewers
the same question (system message ``REVIEWER_SYSTEM_PROMPT``; user message the NeurIPS form, the
answer format, one few-shot paper and review, and the manuscript); each reviewer call is made up to
twice, the second time with ``REVIEW_RETRY_CORRECTION`` appended, and a transport failure is
retried with backoff (six tries); ask the area chair (``META_REVIEWER_SYSTEM_PROMPT``; the form and
the five reviews) once under the same rules; replace the nine numeric scores of the area chair's
answer by the rounded mean of the reviewers'. A review never fails the pipeline or discards the
paper: a failed review is a recorded failure and the paper stays.

The graph::

    record_run_config -> extract_text -> reviewer x N (Send) -> join -> area_chair -> aggregate
        -> publish

The reviewers run in parallel (upstream: one after the other; the calls are identical and
independent). An invalid answer is asked for again in a new node execution, not in a loop inside
the node: a reviewer task sends itself again with the next attempt number, and the area chair step
is a small graph that loops on a counter in state, so that each paid call is checkpointed on its
own and a resume never repeats one. The join is deferred until every reviewer task has finished.
Selecting the review phase is the consent to send the manuscript to the reviewer
models, as it is upstream; the graph asks nothing.

Step names for hooks: ``review.extract_text``, ``review.area_chair``, ``review.aggregate`` and
``review.publish``. The reviewer tasks are the targets of ``Send`` and carry no hook nodes.
"""

from __future__ import annotations

import asyncio
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command, RetryPolicy, Send

from ..agents.factory import is_transient_provider_error
from ..config import ReviewConfig
from ..context import GraphOverrides, ScientistContext
from ..core.render import truncate_utf8
from ..core.results import AIReview, ReviewResult
from ..core.runconfig import method_config, record_run_config
from ..core.state import Event, Failure
from ..core.steps import StepGraph
from ..models import Role
from ..workspace import Workspace
from .models import (
    DEFAULT_ARTIFACT_STEM,
    DEFAULT_MANIFEST_KEY,
    InvalidReviewOutputError,
    ReviewerAnswer,
    ReviewerTask,
    ReviewInput,
    ReviewOutput,
    ReviewState,
    aggregate_review,
    review_payload,
    sorted_reviews,
)
from .pdf import PdfTextError, extract_pdf_text, validate_pdf
from .publish import ManifestError, publish_review
from .texts import (
    area_chair_message,
    area_chair_system_message,
    review_message,
    reviewer_system_message,
    with_retry_correction,
)

FAILURE_DETAIL_BYTES = 1200
"""Upstream bounds a failure detail at 1,200 characters (``_MAX_FAILURE_DETAIL_CHARS``)."""

META_REVIEW = "meta-review"
"""The context name of the area chair call (upstream's ``context``)."""


def reviewer_retry_policy(
    max_attempts: int = 6, initial_interval: float = 1.0, max_interval: float = 16.0
) -> RetryPolicy:
    """Return the retry policy of a model call of the review graph.

    Upstream retries a rate limit, a timeout and a server error with ``backoff.expo`` and at most
    six tries (1, 2, 4, 8, 16 seconds with jitter). The policy retries the transient provider
    failures of :func:`~langgraph_agentic_scientist.agents.factory.is_transient_provider_error`
    in the same way.
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        max_interval=max_interval,
        jitter=True,
        retry_on=is_transient_provider_error,
    )


def _failure_from(error: BaseException, *, context: str) -> Failure:
    detail = truncate_utf8(f"{context}: {type(error).__name__}: {error}", FAILURE_DETAIL_BYTES)
    return Failure(category="review_model_call_failed", scope="infrastructure", detail=detail)


def _invalid_failure(context: str, error: str | None) -> Failure:
    return Failure(
        category="review_output_invalid",
        scope="method",
        detail=truncate_utf8(f"{context}: {error}", FAILURE_DETAIL_BYTES),
    )


def sampling_kwargs(config: ReviewConfig, attempt: int) -> dict[str, Any]:
    """Return the sampling arguments upstream binds to every call.

    ``temperature=1``, ``seed=attempt`` and the completion token limit
    (``max_completion_tokens`` upstream, ``max_tokens`` in LangChain), bound through
    ``Runnable.bind`` so that the structured-output wrapper passes them to the model.
    """
    return {
        "temperature": 1,
        "seed": attempt,
        "max_tokens": config.max_completion_tokens,
    }


def _validated(answer: object, *, context: str) -> AIReview:
    if not isinstance(answer, ReviewerAnswer):
        raise InvalidReviewOutputError(f"{context} did not contain a valid structured review")
    return answer.review


@dataclass
class ReviewAttempt:
    """The outcome of one call to a reviewer role.

    Attributes:
        review: The valid review, or ``None`` when the answer was invalid.
        error: Why the answer was invalid, when ``review`` is ``None``.
    """

    review: AIReview | None = None
    error: str | None = None


async def review_attempt(
    runtime: Runtime[ScientistContext],
    *,
    role: Role,
    system: str,
    message: str,
    config: ReviewConfig,
    attempt: int,
    context: str,
) -> ReviewAttempt:
    """Ask a reviewer role once for a valid review (attempt number ``attempt``, from 0).

    The first attempt sends ``message``; each later one sends it with the retry correction
    appended, as upstream's ``_request_valid_review`` does (the validation error is not added).
    An answer that is missing or violates the review contract (an out-of-range score, a boolean
    as a score, a wrong ``Decision``, a missing field) is invalid; the graph asks again in a new
    node execution, so that every paid call is checkpointed on its own. Transport failures are
    not caught here: the node's retry policy handles them.
    """
    ctx = runtime.context
    call: Runnable[Any, Any] = ctx.agent_factory.structured_call(role, ReviewerAnswer, ctx=ctx)
    text = message if attempt == 0 else with_retry_correction(ctx.prompts, message)
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=text)]
    if ctx.config.review.sampling == "upstream":
        call = call.bind(**sampling_kwargs(config, attempt))
    try:
        return ReviewAttempt(review=_validated(await call.ainvoke(messages), context=context))
    except ValueError as exc:
        return ReviewAttempt(error=str(exc))


def _retry_note(context: str, attempt: int, max_attempts: int) -> str:
    return f"aireview: invalid {context}; retrying {context} (attempt {attempt + 2}/{max_attempts})"


# --- nodes ----------------------------------------------------------------------------------


def _paper_dir(state: Mapping[str, Any]) -> str:
    explicit = state.get("paper_dir")
    if explicit is not None:
        return str(explicit)
    return posixpath.dirname(str(state["pdf_path"]))


async def _extract_text(state: ReviewState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    config = method_config(state).review
    provided = state.get("paper_text")
    if provided is not None:
        if len(provided) < config.min_pdf_text_chars:
            return _extract_failed(
                f"paper text must contain at least {config.min_pdf_text_chars} characters"
            )
        return {"paper_text": provided}
    workspace: Workspace = runtime.context.workspace
    path = state["pdf_path"]
    try:
        data = await workspace.read_bytes(path)
    except FileNotFoundError:
        return _extract_failed(f"PDF does not exist: {path}")
    try:
        validate_pdf(data, path)
        text = await asyncio.to_thread(extract_pdf_text, data, min_chars=config.min_pdf_text_chars)
    except PdfTextError as exc:
        return _extract_failed(str(exc))
    return {"paper_text": text}


def _extract_failed(detail: str) -> dict[str, Any]:
    failure = Failure(category="paper_text_unavailable", scope="task", detail=detail)
    return {"failure": failure, "failures": {"extract": failure}}


def _route_after_extract(state: Mapping[str, Any]) -> Any:
    if state.get("failure") is not None:
        return "publish"
    count = method_config(state).review.num_reviewers
    run_config = state.get("run_config")
    paper_text = state["paper_text"]
    return [
        Send("reviewer", ReviewerTask(index=index, paper_text=paper_text, run_config=run_config))
        for index in range(1, count + 1)
    ]


async def _reviewer(task: ReviewerTask, runtime: Runtime[ScientistContext]) -> Command[Any]:
    """One attempt of one reviewer; an invalid answer sends the task again with the next number."""
    config = method_config(task).review
    ctx = runtime.context
    index = task["index"]
    attempt_number = task.get("attempt", 0)
    notes = list(task.get("notes") or [])
    context = f"review {index}"
    source = f"reviewer {index}"
    message = review_message(ctx.prompts, task["paper_text"], fewshot=config.fewshot)
    attempt = await review_attempt(
        runtime,
        role=Role.AI_REVIEWER,
        system=reviewer_system_message(ctx.prompts),
        message=message,
        config=config,
        attempt=attempt_number,
        context=context,
    )
    if attempt.review is not None:
        update = {"reviews": {str(index): attempt.review}, "notes": {source: notes}}
        return Command(update=update, goto="join_reviews")
    if attempt_number + 1 < config.max_attempts:
        notes.append(_retry_note(context, attempt_number, config.max_attempts))
        retry = ReviewerTask(
            index=index,
            paper_text=task["paper_text"],
            run_config=task.get("run_config"),
            attempt=attempt_number + 1,
            notes=notes,
        )
        return Command(goto=Send("reviewer", retry))
    notes.append(f"aireview: {attempt.error}")
    failure = _invalid_failure(context, attempt.error)
    return Command(
        update={"failures": {source: failure}, "notes": {source: notes}}, goto="join_reviews"
    )


async def _reviewer_failed(task: ReviewerTask, error: NodeError) -> Command[Any]:
    source = f"reviewer {task['index']}"
    failure = _failure_from(error.error, context=f"review {task['index']}")
    return Command(update={"failures": {source: failure}}, goto="join_reviews")


def _join_reviews(state: ReviewState) -> dict[str, Any]:
    count = method_config(state).review.num_reviewers
    reviews = state.get("reviews") or {}
    if len(reviews) == count:
        return {}
    failures = state.get("failures") or {}
    if failures:
        first = failures[sorted(failures)[0]]
        return {"failure": first}
    return {
        "failure": Failure(
            category="review_output_invalid",
            scope="method",
            detail=f"{len(reviews)} of {count} reviews were produced",
        )
    }


def _route_after_join(state: Mapping[str, Any]) -> str:
    return "failed" if state.get("failure") is not None else "complete"


async def _area_chair_attempt(
    state: ReviewState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """One attempt of the area chair; the counter and the log lines are kept in state."""
    config = method_config(state).review
    ctx = runtime.context
    reviews = sorted_reviews(state["reviews"])
    attempt_number = state.get("meta_attempt") or 0
    notes = list((state.get("notes") or {}).get(META_REVIEW) or [])
    attempt = await review_attempt(
        runtime,
        role=Role.AREA_CHAIR,
        system=area_chair_system_message(ctx.prompts, len(reviews)),
        message=area_chair_message(ctx.prompts, reviews),
        config=config,
        attempt=attempt_number,
        context=META_REVIEW,
    )
    if attempt.review is not None:
        return {"meta": attempt.review, "meta_error": None, "notes": {META_REVIEW: notes}}
    if attempt_number + 1 < config.max_attempts:
        notes.append(_retry_note(META_REVIEW, attempt_number, config.max_attempts))
    else:
        notes.append(f"aireview: {attempt.error}")
    return {
        "meta_attempt": attempt_number + 1,
        "meta_error": attempt.error,
        "notes": {META_REVIEW: notes},
    }


def _route_area_chair_attempt(state: ReviewState) -> str:
    if state.get("meta") is not None:
        return "done"
    if (state.get("meta_attempt") or 0) < method_config(state).review.max_attempts:
        return "again"
    return "invalid"


def _area_chair_invalid(state: ReviewState) -> dict[str, Any]:
    failure = _invalid_failure(META_REVIEW, state.get("meta_error"))
    return {"failure": failure, "failures": {META_REVIEW: failure}}


def _build_area_chair_graph(
    policy: RetryPolicy,
) -> CompiledStateGraph[ReviewState, ScientistContext, ReviewState, ReviewState]:
    """Build the area chair step: one call per node execution, asked again while invalid."""
    graph: StateGraph[ReviewState, ScientistContext, ReviewState, ReviewState] = StateGraph(
        ReviewState, context_schema=ScientistContext
    )
    graph.add_node("attempt", _area_chair_attempt, retry_policy=policy)
    graph.add_node("invalid", _area_chair_invalid)
    graph.add_edge(START, "attempt")
    graph.add_conditional_edges(
        "attempt",
        _route_area_chair_attempt,
        {"done": END, "again": "attempt", "invalid": "invalid"},
    )
    graph.add_edge("invalid", END)
    return graph.compile(name="area_chair")


def _area_chair_failed(publish_entry: str) -> Any:
    """Return the error handler of the area chair step: record the failure, go to ``publish``."""

    async def handle(state: ReviewState, error: NodeError) -> Command[Any]:
        failure = _failure_from(error.error, context=META_REVIEW)
        update = {"failure": failure, "failures": {META_REVIEW: failure}}
        return Command(update=update, goto=publish_entry)

    return handle


def _route_after_area_chair(state: Mapping[str, Any]) -> str:
    return "failed" if state.get("failure") is not None or state.get("meta") is None else "ok"


def _aggregate(state: ReviewState) -> dict[str, Any]:
    meta = state["meta"]
    assert meta is not None
    return {"aggregate": aggregate_review(meta, sorted_reviews(state["reviews"]))}


def _source_order(source: str) -> tuple[int, int]:
    if source == META_REVIEW:
        return (1, 0)
    return (0, int(source.rsplit(" ", 1)[-1]))


def _log_text(state: ReviewState) -> str:
    notes = state.get("notes") or {}
    ordered = sorted(notes, key=_source_order)
    return "".join(f"{line}\n" for source in ordered for line in notes[source])


async def _publish(state: ReviewState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    workspace: Workspace = runtime.context.workspace
    paper_dir = _paper_dir(state)
    stem = state.get("artifact_stem") or DEFAULT_ARTIFACT_STEM
    key = state.get("manifest_key") or DEFAULT_MANIFEST_KEY
    aggregate = state.get("aggregate")
    failure = state.get("failure")
    log_text = _log_text(state)
    reviews = sorted_reviews(state.get("reviews") or {})
    path: str | None = None
    if aggregate is not None and failure is None:
        try:
            path = await publish_review(
                workspace,
                paper_dir=paper_dir,
                artifact_stem=stem,
                manifest_key=key,
                status="completed",
                payload=review_payload(aggregate),
                log_text=log_text,
                error=None,
            )
        except ManifestError as exc:
            failure = Failure(category="paper_manifest_invalid", scope="task", detail=str(exc))
    if path is None:
        failure = failure or Failure(category="review_failed", detail="the review failed")
        try:
            await publish_review(
                workspace,
                paper_dir=paper_dir,
                artifact_stem=stem,
                manifest_key=key,
                status="failed",
                payload=None,
                log_text=log_text,
                error=failure.detail or failure.category,
            )
        except ManifestError as exc:
            failure = Failure(category="paper_manifest_invalid", scope="task", detail=str(exc))
            await publish_review(
                workspace,
                paper_dir=paper_dir,
                artifact_stem=stem,
                manifest_key=key,
                status="failed",
                payload=None,
                log_text=log_text,
                error=failure.detail,
                update_manifest=False,
            )
    result = ReviewResult(
        status="completed" if path is not None else "failed",
        reviews=reviews,
        meta_review=state.get("meta"),
        aggregate=aggregate if path is not None else None,
        review_path=path,
        failure=None if path is not None else failure,
    )
    data: dict[str, Any] = {"status": result.status, "review_path": path}
    if aggregate is not None and path is not None:
        data.update(overall_score=aggregate.overall, decision=aggregate.decision)
    event = Event(kind="review_result", phase="review", step="publish", data=data)
    return {"result": result, "events": [event]}


def build_review_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "review",
    retry_policy: RetryPolicy | None = None,
) -> CompiledStateGraph[ReviewState, ScientistContext, ReviewInput, ReviewOutput]:
    """Build the AI review graph.

    The input is a :class:`~langgraph_agentic_scientist.review.models.ReviewInput` (the PDF path
    in the workspace, optionally the artifact name and the manuscript text); the output is a
    :class:`~langgraph_agentic_scientist.core.results.ReviewResult` under ``result``. The roles
    ``Role.AI_REVIEWER`` (five calls) and ``Role.AREA_CHAIR`` (one call) choose the models; the
    structured-output method of each comes from the model registry. The review settings are
    those of ``ScientistConfig.review`` (reviewer count, attempts, token limit, minimum PDF text,
    few-shot example, sampling).

    Args:
        checkpointer: The checkpointer, when the graph is the top level graph.
        store: The store, when the graph is the top level graph.
        overrides: Replacement graphs (this graph has no slot of its own; hooks are the
            extension point of its steps).
        name: The graph name (traces nest by it).
        retry_policy: The retry policy of the model-calling nodes; the default is upstream's six
            tries with exponential backoff (:func:`reviewer_retry_policy`).

    Returns:
        The compiled graph; use ``ainvoke`` (all nodes are asynchronous).
    """
    policy = retry_policy or reviewer_retry_policy()
    sg = StepGraph("review", ReviewState, input_schema=ReviewInput, output_schema=ReviewOutput)
    sg.add_node("record_run_config", record_run_config)
    sg.step("extract_text", _extract_text, hook_writable=("paper_text",))
    sg.add_node(
        "reviewer",
        _reviewer,
        retry_policy=policy,
        error_handler=_reviewer_failed,
        destinations=("reviewer", "join_reviews"),
    )
    sg.add_node("join_reviews", _join_reviews, defer=True)
    sg.step("publish", _publish)
    sg.step(
        "area_chair",
        _build_area_chair_graph(policy),
        hook_writable=("meta",),
        error_handler=_area_chair_failed(sg.entry("publish")),
    )
    sg.step("aggregate", _aggregate, hook_writable=("aggregate",))

    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "extract_text")
    sg.add_conditional_edges("extract_text", _route_after_extract)
    sg.add_conditional_edges(
        "join_reviews", _route_after_join, {"complete": "area_chair", "failed": "publish"}
    )
    sg.add_conditional_edges(
        "area_chair", _route_after_area_chair, {"ok": "aggregate", "failed": "publish"}
    )
    sg.add_edge("aggregate", "publish")
    sg.add_edge("publish", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "ReviewAttempt",
    "build_review_graph",
    "review_attempt",
    "reviewer_retry_policy",
    "sampling_kwargs",
]
