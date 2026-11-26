"""The review graph alone, with scripted models: the ensemble, retries, sampling and publication."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from review_helpers import PAPER_TEXT, answer, last_text, make_pdf, review_dict
from testkit import fast_config

from langgraph_agentic_scientist import (
    Hooks,
    LocalWorkspace,
    ReviewConfig,
    ScientistConfig,
    ScientistContext,
)
from langgraph_agentic_scientist.core.results import AIReview, ReviewResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.review import build_review_graph, reviewer_retry_policy
from langgraph_agentic_scientist.review.models import SCORE_KEYS
from langgraph_agentic_scientist.testing import ScriptedChatModel, ScriptedModels

RETRY_TEXT = "Retry correction: The previous response did not satisfy the required review JSON"
FAST = reviewer_retry_policy(initial_interval=0.001, max_interval=0.002)


def _config(**review: Any) -> ScientistConfig:
    return fast_config(review=ReviewConfig(**review))


async def _run(
    ctx: ScientistContext,
    *,
    graph: Any = None,
    thread: str = "t",
    **inputs: Any,
) -> dict[str, Any]:
    graph = graph or build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    payload = {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT, **inputs}
    return await graph.ainvoke(payload, {"configurable": {"thread_id": thread}}, context=ctx)


def _model(ctx: ScientistContext, role: Role) -> ScriptedChatModel:
    assert isinstance(ctx.models, ScriptedModels)
    return ctx.models.scripted(role)


def _kwargs(model: ScriptedChatModel) -> list[dict[str, Any]]:
    return model._state.kwargs


def _written(workspace: LocalWorkspace, name: str) -> dict[str, Any]:
    return json.loads((workspace_root(workspace) / "paper" / name).read_text(encoding="utf-8"))


def workspace_root(workspace: LocalWorkspace) -> Any:
    return workspace.root


def _by_retry(valid: AIMessage, invalid: AIMessage) -> Callable[[list[BaseMessage]], AIMessage]:
    def respond(messages: list[BaseMessage]) -> AIMessage:
        return valid if RETRY_TEXT in last_text(messages) else invalid

    return respond


# --- the ensemble ---------------------------------------------------------------------------


async def test_five_reviewers_and_an_area_chair_produce_the_mean_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    overall = [4, 5, 5, 6, 7]
    reviewers = [answer(review_dict(Overall=value, Confidence=3 + value % 2)) for value in overall]
    chair = answer(review_dict(Summary="Consensus.", Overall=9, Decision="Accept"))
    ctx = make_context({Role.AI_REVIEWER: reviewers, Role.AREA_CHAIR: [chair]})

    out = await _run(ctx)

    result: ReviewResult = out["result"]
    assert result.status == "completed"
    assert result.failure is None
    assert len(result.reviews) == 5
    assert result.review_path == "paper/ai_review.json"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 5
    assert _model(ctx, Role.AREA_CHAIR).call_count == 1
    written = _written(workspace, "ai_review.json")
    assert written["Overall"] == 5  # mean 5.4
    assert written["Summary"] == "Consensus."
    assert written["Decision"] == "Accept"
    assert "Ethical Concerns" in written
    assert all(isinstance(written[key], int) for key in SCORE_KEYS)
    assert (workspace.root / "paper" / "ai_review.log").is_file()
    assert result.meta_review is not None
    assert result.meta_review.overall == 9  # the chair's own answer is kept in the result
    assert result.aggregate is not None
    assert result.aggregate.overall == 5


async def test_the_reviewers_run_in_parallel(
    make_context: Callable[..., ScientistContext],
) -> None:
    barrier = threading.Barrier(5, timeout=20)

    def reviewer(messages: list[BaseMessage]) -> AIMessage:
        barrier.wait()  # every reviewer call must be in flight at the same time
        return answer()

    ctx = make_context({Role.AI_REVIEWER: [reviewer] * 5, Role.AREA_CHAIR: [answer()]})
    out = await _run(ctx)
    assert out["result"].status == "completed"


async def test_the_reviewer_and_the_area_chair_get_upstreams_messages(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()] * 5, Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=5),
    )
    await _run(ctx)
    reviewer_call = _model(ctx, Role.AI_REVIEWER).calls[0]
    assert [message.type for message in reviewer_call] == ["system", "human"]
    assert reviewer_call[0].content == ctx.prompts.text("review/reviewer_system")
    user = last_text(reviewer_call)
    assert user.startswith(ctx.prompts.text("review/neurips_form"))
    assert (
        "Below are some sample reviews, copied from previous machine learning conferences." in user
    )
    assert user.endswith("Here is the paper you are asked to review:\n```\n" + PAPER_TEXT + "\n```")
    chair_call = _model(ctx, Role.AREA_CHAIR).calls[0]
    assert "reviewed by 5 reviewers" in str(chair_call[0].content)
    chair_user = last_text(chair_call)
    assert "Review 1/5:\n```\n" in chair_user
    assert "Review 5/5:\n```\n" in chair_user
    assert '"Ethical Concerns": false' in chair_user
    assert "sample reviews" not in chair_user


async def test_the_few_shot_example_can_be_left_out(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1, fewshot=False),
    )
    await _run(ctx)
    user = last_text(_model(ctx, Role.AI_REVIEWER).calls[0])
    assert "sample reviews" not in user
    assert len(user) < 20_000


async def test_the_reviewer_count_is_a_setting(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()] * 3, Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=3),
    )
    out = await _run(ctx)
    assert len(out["result"].reviews) == 3
    assert "reviewed by 3 reviewers" in str(_model(ctx, Role.AREA_CHAIR).calls[0][0].content)


async def test_two_reviewers_with_a_half_mean_round_to_the_even_score(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = make_context(
        {
            Role.AI_REVIEWER: [answer(review_dict(Overall=2)), answer(review_dict(Overall=3))],
            Role.AREA_CHAIR: [answer(review_dict(Overall=10))],
        },
        config=_config(num_reviewers=2),
    )
    await _run(ctx)
    assert _written(workspace, "ai_review.json")["Overall"] == 2  # 2.5 -> 2

    ctx = make_context(
        {
            Role.AI_REVIEWER: [answer(review_dict(Overall=3)), answer(review_dict(Overall=4))],
            Role.AREA_CHAIR: [answer(review_dict(Overall=1))],
        },
        config=_config(num_reviewers=2),
    )
    await _run(ctx, thread="second")
    assert _written(workspace, "ai_review.json")["Overall"] == 4  # 3.5 -> 4


# --- retries on invalid answers -----------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        review_dict(Overall=11),
        review_dict(Originality=5),
        review_dict(Confidence=0),
        review_dict(Decision="Weak Accept"),
        review_dict(Quality=True),
        review_dict(**{"Ethical Concerns": "no"}),
        {key: value for key, value in review_dict().items() if key != "Summary"},
        {**review_dict(), "Strengths": ["fine", 3]},
    ],
    ids=[
        "overall-above-range",
        "originality-above-range",
        "confidence-below-range",
        "weak-accept",
        "boolean-score",
        "ethical-concerns-not-boolean",
        "missing-field",
        "list-of-non-strings",
    ],
)
async def test_an_invalid_first_answer_is_asked_for_again_with_the_retry_correction(
    make_context: Callable[..., ScientistContext], bad: Mapping[str, Any]
) -> None:
    respond = _by_retry(answer(), answer(dict(bad)))
    ctx = make_context(
        {Role.AI_REVIEWER: [respond] * 2, Role.AREA_CHAIR: [respond] * 2},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)

    assert out["result"].status == "completed"
    reviewer_calls = _model(ctx, Role.AI_REVIEWER).calls
    assert len(reviewer_calls) == 2
    assert RETRY_TEXT not in last_text(reviewer_calls[0])
    assert last_text(reviewer_calls[1]).endswith(ctx.prompts.text("review/retry_correction"))
    assert last_text(reviewer_calls[1]).startswith(last_text(reviewer_calls[0]))
    # the area chair's call is asked for again under the same rules
    assert len(_model(ctx, Role.AREA_CHAIR).calls) == 2


async def test_two_invalid_answers_fail_the_review_and_the_paper_stays(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    paper = workspace.root / "paper"
    paper.mkdir(parents=True)
    (paper / "paper.pdf").write_bytes(b"%PDF-1.7 the paper")
    (paper / "ai_review.json").write_text('{"stale": true}', encoding="utf-8")
    bad = answer(review_dict(Overall=11))
    ctx = make_context(
        {Role.AI_REVIEWER: [bad] * 2, Role.AREA_CHAIR: []}, config=_config(num_reviewers=1)
    )

    out = await _run(ctx)

    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "review_output_invalid"
    assert "review 1" in result.failure.detail
    assert _model(ctx, Role.AI_REVIEWER).call_count == 2  # at most two attempts
    assert _model(ctx, Role.AREA_CHAIR).call_count == 0
    assert (paper / "paper.pdf").is_file()
    assert not (paper / "ai_review.json").exists()  # a failed review clears the stale copy
    log = (paper / "ai_review.log").read_text(encoding="utf-8")
    assert "aireview: invalid review 1; retrying review 1 (attempt 2/2)" in log
    assert "[ai review failure]" in log


async def test_the_area_chairs_invalid_answers_fail_the_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = make_context(
        {
            Role.AI_REVIEWER: [answer()],
            Role.AREA_CHAIR: [answer(review_dict(Decision="Maybe"))] * 2,
        },
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert "meta-review" in result.failure.detail
    assert len(result.reviews) == 1
    assert not (workspace.root / "paper" / "ai_review.json").exists()


async def test_the_number_of_attempts_is_a_setting(
    make_context: Callable[..., ScientistContext],
) -> None:
    bad = answer(review_dict(Overall=0))
    ctx = make_context(
        {Role.AI_REVIEWER: [bad] * 3, Role.AREA_CHAIR: []},
        config=_config(num_reviewers=1, max_attempts=3),
    )
    out = await _run(ctx)
    assert out["result"].status == "failed"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 3


# --- sampling ---------------------------------------------------------------------------------


async def test_the_calls_bind_upstreams_sampling(
    make_context: Callable[..., ScientistContext],
) -> None:
    respond = _by_retry(answer(), answer(review_dict(Overall=0)))
    ctx = make_context(
        {Role.AI_REVIEWER: [respond] * 2, Role.AREA_CHAIR: [respond] * 2},
        config=_config(num_reviewers=1),
    )
    await _run(ctx)
    for role in (Role.AI_REVIEWER, Role.AREA_CHAIR):
        first, second = _kwargs(_model(ctx, role))
        for attempt, kwargs in enumerate((first, second)):
            assert kwargs["temperature"] == 1
            assert kwargs["seed"] == attempt  # seed = attempt
            assert kwargs["max_tokens"] == 16384


async def test_the_token_limit_is_a_setting_and_sampling_can_be_left_to_the_model(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1, max_completion_tokens=2048),
    )
    await _run(ctx)
    assert _kwargs(_model(ctx, Role.AI_REVIEWER))[0]["max_tokens"] == 2048

    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1, sampling="none"),
    )
    await _run(ctx, thread="none")
    for role in (Role.AI_REVIEWER, Role.AREA_CHAIR):
        kwargs = _kwargs(_model(ctx, role))[0]
        assert "temperature" not in kwargs
        assert "seed" not in kwargs
        assert "max_tokens" not in kwargs


# --- transport failures -----------------------------------------------------------------------


async def test_a_transport_failure_is_retried_by_the_node_policy(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {
            Role.AI_REVIEWER: [ConnectionError("reset"), ConnectionError("reset"), answer()],
            Role.AREA_CHAIR: [TimeoutError("slow"), answer()],
        },
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    assert out["result"].status == "completed"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 3
    assert _model(ctx, Role.AREA_CHAIR).call_count == 2


async def test_six_failed_transports_end_the_review_as_a_failure_not_a_crash(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [ConnectionError("down")] * 8, Role.AREA_CHAIR: []},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "review_model_call_failed"
    assert result.failure.scope == "infrastructure"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 6  # upstream: six tries
    assert _model(ctx, Role.AREA_CHAIR).call_count == 0


async def test_a_failure_that_is_not_transient_is_not_retried(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [PermissionError("401")] * 3, Role.AREA_CHAIR: []},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    assert out["result"].status == "failed"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 1


# --- no consent question ----------------------------------------------------------------------


class _NoQuestions:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def decide(self, question: Any) -> None:
        self.asked.append(question)


async def test_selecting_the_phase_is_the_consent_so_nothing_is_asked(
    make_context: Callable[..., ScientistContext],
) -> None:
    policy = _NoQuestions()
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()] * 5, Role.AREA_CHAIR: [answer()]}, policy=policy
    )
    out = await _run(ctx)
    assert out["result"].status == "completed"
    assert "__interrupt__" not in out
    assert policy.asked == []


# --- the PDF ----------------------------------------------------------------------------------


async def test_the_manuscript_text_comes_from_the_pdf(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    pytest.importorskip("pymupdf")
    (workspace.root / "paper").mkdir(parents=True)
    (workspace.root / "paper" / "paper.pdf").write_bytes(make_pdf())
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    graph = build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    out = await graph.ainvoke(
        {"pdf_path": "paper/paper.pdf"}, {"configurable": {"thread_id": "pdf"}}, context=ctx
    )
    assert out["result"].status == "completed"
    assert "relationship between circle packing and search" in last_text(
        _model(ctx, Role.AI_REVIEWER).calls[0]
    )


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        (None, "PDF does not exist"),
        (b"not a pdf at all", "valid PDF header"),
        (b"", "valid PDF header"),
    ],
)
async def test_an_unreadable_pdf_fails_the_review_without_calling_a_model(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    data: bytes | None,
    needle: str,
) -> None:
    (workspace.root / "paper").mkdir(parents=True)
    if data is not None:
        (workspace.root / "paper" / "paper.pdf").write_bytes(data)
    ctx = make_context({Role.AI_REVIEWER: [], Role.AREA_CHAIR: []})
    graph = build_review_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(
        {"pdf_path": "paper/paper.pdf"}, {"configurable": {"thread_id": "bad"}}, context=ctx
    )
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "paper_text_unavailable"
    assert needle in result.failure.detail
    assert _model(ctx, Role.AI_REVIEWER).call_count == 0


async def test_a_pdf_without_enough_text_fails_the_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    pytest.importorskip("pymupdf")
    (workspace.root / "paper").mkdir(parents=True)
    (workspace.root / "paper" / "paper.pdf").write_bytes(make_pdf("Too short."))
    ctx = make_context({Role.AI_REVIEWER: [], Role.AREA_CHAIR: []})
    graph = build_review_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(
        {"pdf_path": "paper/paper.pdf"}, {"configurable": {"thread_id": "short"}}, context=ctx
    )
    assert out["result"].status == "failed"
    assert "at least 100 characters" in out["result"].failure.detail


async def test_given_text_must_also_reach_the_minimum(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context({Role.AI_REVIEWER: [], Role.AREA_CHAIR: []})
    out = await _run(ctx, paper_text="short")
    assert out["result"].status == "failed"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 0


# --- publication ------------------------------------------------------------------------------


async def test_the_artifacts_are_named_by_the_stem_and_the_manifest_is_updated(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    paper = workspace.root / "paper"
    paper.mkdir(parents=True)
    (paper / "latest_manifest.json").write_text(
        json.dumps({"paper_pdf": "paper/paper.pdf", "keep": 1}), encoding="utf-8"
    )
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx, artifact_stem="ai_review.before_revision", manifest_key="before_revision")
    assert out["result"].review_path == "paper/ai_review.before_revision.json"
    for name in (
        "ai_review.before_revision.json",
        "ai_review.before_revision.log",
        "ai_review.json",
        "ai_review.log",
    ):
        assert (paper / name).is_file()
    assert (paper / "ai_review.json").read_text() == (
        paper / "ai_review.before_revision.json"
    ).read_text()
    manifest = json.loads((paper / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["keep"] == 1
    assert manifest["before_revision"] == {
        "status": "completed",
        "review_json": "paper/ai_review.before_revision.json",
        "log": "paper/ai_review.before_revision.log",
    }
    assert manifest["ai_review"]["review_json"] == "paper/ai_review.json"


async def test_a_failed_review_is_recorded_in_the_manifest(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    paper = workspace.root / "paper"
    paper.mkdir(parents=True)
    (paper / "latest_manifest.json").write_text("{}", encoding="utf-8")
    ctx = make_context({Role.AI_REVIEWER: [answer(review_dict(Overall=0))] * 2})
    out = await _run(ctx, paper_text="short")
    assert out["result"].status == "failed"
    manifest = json.loads((paper / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ai_review"]["status"] == "failed"
    assert manifest["ai_review"]["review_json"] is None


async def test_an_unusable_manifest_fails_the_review_but_writes_the_log(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    paper = workspace.root / "paper"
    paper.mkdir(parents=True)
    (paper / "latest_manifest.json").write_text("[1, 2]", encoding="utf-8")
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "paper_manifest_invalid"
    assert not (paper / "ai_review.json").exists()
    assert "paper manifest must contain a JSON object" in (paper / "ai_review.log").read_text()


async def test_a_review_outside_a_pipeline_needs_no_manifest(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx, paper_dir="elsewhere")
    assert out["result"].review_path == "elsewhere/ai_review.json"
    assert (workspace.root / "elsewhere" / "ai_review.json").is_file()
    assert not (workspace.root / "elsewhere" / "latest_manifest.json").exists()


async def test_the_result_event_names_the_score_and_the_decision(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer(review_dict(Overall=4))], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    graph = build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    await graph.ainvoke(
        {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT},
        {"configurable": {"thread_id": "events"}},
        context=ctx,
    )
    state = await graph.aget_state({"configurable": {"thread_id": "events"}})
    events = [event for event in state.values["events"] if event.kind == "review_result"]
    assert events[0].data["overall_score"] == 4
    assert events[0].data["decision"] == "Reject"


# --- hooks, resume, run configuration -------------------------------------------------------


async def test_hooks_fire_at_the_review_steps(
    make_context: Callable[..., ScientistContext],
) -> None:
    seen: list[str] = []

    def hook(name: str) -> Callable[..., Any]:
        async def run(state: Mapping[str, Any], runtime: Any) -> None:
            seen.append(name)

        return run

    steps = ("extract_text", "area_chair", "aggregate", "publish")
    hooks = Hooks(
        before={f"review.{step}": [hook(f"before {step}")] for step in steps},
        after={f"review.{step}": [hook(f"after {step}")] for step in steps},
    )
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
        hooks=hooks,
    )
    await _run(ctx)
    assert seen == [f"{when} {step}" for step in steps for when in ("before", "after")]


async def test_a_resume_after_a_crash_does_not_ask_the_reviewers_again(
    make_context: Callable[..., ScientistContext],
) -> None:
    crashes = {"left": 1}

    async def crash_once(state: Mapping[str, Any], runtime: Any) -> None:
        if crashes["left"]:
            crashes["left"] -= 1
            raise RuntimeError("the host died")

    ctx = make_context(
        {Role.AI_REVIEWER: [answer()] * 5, Role.AREA_CHAIR: [answer()]},
        hooks=Hooks(before={"review.area_chair": [crash_once]}),
    )
    graph = build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    config = {"configurable": {"thread_id": "crash"}}
    inputs = {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT}
    with pytest.raises(RuntimeError, match="host died"):
        await graph.ainvoke(inputs, config, context=ctx)
    assert _model(ctx, Role.AI_REVIEWER).call_count == 5

    out = await graph.ainvoke(None, config, context=ctx)

    assert out["result"].status == "completed"
    assert _model(ctx, Role.AI_REVIEWER).call_count == 5  # no reviewer asked twice
    assert _model(ctx, Role.AREA_CHAIR).call_count == 1


async def test_each_attempt_is_checkpointed_so_a_resume_does_not_repeat_it(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {
            Role.AI_REVIEWER: [answer(review_dict(Overall=11)), answer(), answer()],
            Role.AREA_CHAIR: [answer(), answer()],
        },
        config=_config(num_reviewers=1),
    )
    graph = build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    config: Any = {"configurable": {"thread_id": "attempts"}}
    inputs = {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT}
    assert (await graph.ainvoke(inputs, config, context=ctx))["result"].status == "completed"

    # The checkpoints whose next task is a reviewer: before the first attempt and, newer, after
    # the invalid first attempt. A crash at the newer one resumes with the second attempt.
    pending = [
        snapshot
        async for snapshot in graph.aget_state_history(config)
        if snapshot.next == ("reviewer",)
    ]
    assert len(pending) == 2
    out = await graph.ainvoke(None, pending[0].config, context=ctx)

    assert out["result"].status == "completed"
    first_attempts = [
        call for call in _model(ctx, Role.AI_REVIEWER).calls if RETRY_TEXT not in last_text(call)
    ]
    assert len(first_attempts) == 1  # the paid first attempt was not made again
    assert _model(ctx, Role.AI_REVIEWER).call_count == 3


async def test_the_run_configuration_is_recorded_and_a_changed_setting_is_refused(
    make_context: Callable[..., ScientistContext],
) -> None:
    from langgraph_agentic_scientist import ConfigurationMismatch

    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    graph = build_review_graph(checkpointer=InMemorySaver(), retry_policy=FAST)
    config = {"configurable": {"thread_id": "cfg"}}
    await graph.ainvoke(
        {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT}, config, context=ctx
    )
    state = await graph.aget_state(config)
    assert state.values["run_config"].method_config["review"]["num_reviewers"] == 1

    other = make_context(
        {Role.AI_REVIEWER: [], Role.AREA_CHAIR: []}, config=_config(num_reviewers=2)
    )
    with pytest.raises(ConfigurationMismatch, match=r"review\.num_reviewers"):
        await graph.ainvoke(
            {"pdf_path": "paper/paper.pdf", "paper_text": PAPER_TEXT},
            config,
            context=other,
        )


async def test_the_review_is_an_ordinary_review_record(
    make_context: Callable[..., ScientistContext],
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    assert all(isinstance(review, AIReview) for review in out["result"].reviews)


async def test_reviewers_that_fail_do_not_stop_the_others_and_the_review_fails_once(
    make_context: Callable[..., ScientistContext],
) -> None:
    def respond(messages: list[BaseMessage]) -> AIMessage:
        return answer()

    # five reviewers share one script of six transport failures and four answers: every reviewer
    # fails once and is retried, four of them then get an answer, and the fifth finds the script
    # exhausted, which is a failure that is not transient
    failing = [ConnectionError("down")] * 6
    ctx = make_context(
        {Role.AI_REVIEWER: [*failing, respond, respond, respond, respond], Role.AREA_CHAIR: []},
        config=_config(num_reviewers=5),
    )
    out = await _run(ctx)
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "review_model_call_failed"
    assert len(result.reviews) == 4
    assert _model(ctx, Role.AREA_CHAIR).call_count == 0


async def test_an_area_chair_that_cannot_be_reached_fails_the_review_and_keeps_the_reviews(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    ctx = make_context(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [ConnectionError("down")] * 6},
        config=_config(num_reviewers=1),
    )
    out = await _run(ctx)
    result: ReviewResult = out["result"]
    assert result.status == "failed"
    assert result.failure is not None
    assert "meta-review" in result.failure.detail
    assert len(result.reviews) == 1
    assert result.aggregate is None
    assert _model(ctx, Role.AREA_CHAIR).call_count == 6
    assert not (workspace.root / "paper" / "ai_review.json").exists()


async def test_each_role_uses_its_own_structured_output_method(
    make_context: Callable[..., ScientistContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph_agentic_scientist.agents import factory

    methods: dict[str, str] = {}
    real = factory.structured_output

    def record(model: Any, schema: Any, method: str) -> Any:
        methods[model.label] = method  # the scripted model's label names its role
        return real(model, schema, "function_calling")

    monkeypatch.setattr(factory, "structured_output", record)
    models = ScriptedModels.from_scripts(
        {Role.AI_REVIEWER: [answer()], Role.AREA_CHAIR: [answer()]},
        context_window=200_000,
        structured_output_by_role={Role.AI_REVIEWER: "json_schema"},
    )
    ctx = make_context(models=models, config=_config(num_reviewers=1))
    out = await _run(ctx)
    assert out["result"].status == "completed"
    assert methods == {"role ai_reviewer": "json_schema", "role area_chair": "function_calling"}
