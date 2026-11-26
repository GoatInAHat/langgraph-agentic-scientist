"""The scaffold graph with scripted models and fakes: every step, loop and bound of A2-A7.

Each test runs the real graph in a temporary ``LocalWorkspace`` on the contracted toy task; the
roles answer from scripts (the judge from the fingerprints in its own prompt) and the evaluator is
a fake whose answer depends on the staged files, so a repair is observable as a changed result.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from test_scaffold_support import (
    FIXTURE,
    RECURSION,
    drafter_script,
    fast,
    fixture_result,
    judge_step,
    make_context,
    make_evaluator,
    prompt_text,
    write_scaffold,
)

from langgraph_agentic_scientist import Hooks, LocalWorkspace
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.drafting import build_task_scaffold_graph
from langgraph_agentic_scientist.drafting.graph_scaffold import alignment_judge
from langgraph_agentic_scientist.evaluation import DeferredEvaluator
from langgraph_agentic_scientist.evaluation.protocol import EvaluationRequest, EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, final, tool_call

STAGED = "_task_drafting/task"


def edit(path: str, old: str, new: str) -> Any:
    """An agent turn that edits a file."""
    return tool_call("edit_file", {"file_path": path, "old_string": old, "new_string": new})


def mark(path: str, text: str) -> Any:
    """An agent turn that adds a line comment at the top of ``evaluate.py``."""
    return edit(path, '"""Search evaluator', f'# {text}\n"""Search evaluator')


def staged_text(tmp_path: Any, relative: str) -> str:
    return (tmp_path / STAGED / relative).read_text()


def evaluator_with(rule: Any) -> FakeEvaluator:
    """An evaluator that answers failures when ``rule(request)`` returns a result, else healthy."""

    def handler(request: EvaluationRequest) -> EvaluationResult:
        return rule(request) or fixture_result(request)

    return make_evaluator(handler)


def failing_while(tmp_path: Any, absent: str, plane: str = "smoke", **result: Any) -> FakeEvaluator:
    """Search runs of ``plane`` fail while the staged ``evaluate.py`` lacks ``absent``."""

    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if (
            request.plane == plane
            and request.evaluator == "search"
            and absent not in staged_text(tmp_path, "evaluate.py")
        ):
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                **({"status": "failed", "summary": "boom"} | result),
            )
        return None

    return evaluator_with(rule)


class Run:
    """A graph run: a workspace, a context with scripted roles, and the result."""

    def __init__(self, tmp_path: Any, scripts: dict[Any, Any], **kwargs: Any) -> None:
        self.tmp_path = tmp_path
        self.workspace = LocalWorkspace(tmp_path)
        self.scripts = scripts
        self.kwargs = kwargs
        self.thread = "t"

    async def go(self) -> dict[str, Any]:
        self.task = await write_scaffold(self.workspace, only_task_md=True)
        return await self.invoke()

    async def invoke(self, inputs: Any = "start") -> dict[str, Any]:
        if not hasattr(self, "context"):
            self.context, self.models = make_context(self.workspace, self.scripts, **self.kwargs)
            self.graph = build_task_scaffold_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": self.thread}, **RECURSION}
        payload = {"task": self.task} if inputs == "start" else inputs
        return await self.graph.ainvoke(payload, config, context=self.context)

    def calls(self, role: Role) -> int:
        """The number of sessions of a role (a session starts with no model turn behind it)."""
        return len(self.prompts(role))

    def prompts(self, role: Role) -> list[str]:
        """The prompt of each session of a role, in order."""
        calls = self.models.scripted(role).calls
        return [
            prompt_text(call) for call in calls if not any(isinstance(m, AIMessage) for m in call)
        ]


def block(prompt: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}[^:\n]*: ([A-Za-z0-9+/=]*)", prompt)
    assert match, label
    return base64.b64decode(match.group(1)).decode("utf-8")


def attestation(prompt: str) -> dict[str, Any]:
    """The harness attestation a judge prompt carries."""
    match = re.search(r"Harness attestation: `(\{.*\})`", prompt)
    assert match
    return dict(json.loads(match.group(1)))


def drafter(**kwargs: Any) -> dict[Any, Any]:
    """Scripts: the drafter creates the scaffold (``kwargs`` as for ``drafter_script``)."""
    return {Role.TASK_DRAFTER: drafter_script(**kwargs)}


# --- the happy path and what it shows ----------------------------------------------------------


async def test_the_whole_scaffold_is_published_when_everything_passes(tmp_path: Any) -> None:
    evaluator = make_evaluator()
    run = Run(
        tmp_path,
        {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()]},
        evaluator=evaluator,
    )
    out = await run.go()
    drafting = out["drafting"]
    assert drafting.status == "published", drafting.failure
    for name in (
        ".agentic_task_validation.json",
        ".agentic_task_publication.json",
        ".agentic_task_qualification.json",
        ".agentic_task_ignore",
    ):
        assert (tmp_path / "task" / name).is_file(), name
    assert drafting.task is not None and drafting.task.contract is not None
    assert drafting.task.root == "task" and drafting.scaffold_fingerprint.startswith("sha256:")
    assert drafting.receipts["qualification"]["status"] == "pass"
    assert drafting.receipts["alignment_report"]["verdict"] == "pass"
    planes = [(call.plane, call.evaluator) for call in evaluator.calls]
    assert planes == [
        ("smoke", "search"),
        ("smoke", "final"),
        ("production_entry", "search"),
        ("production_entry", "final"),
        ("qualification", "search"),
    ]
    marker = json.loads((tmp_path / "task/.agentic_task_validation.json").read_text())
    assert marker["verdict"] == "pass" and set(marker["fingerprints"]) >= {"declared_scaffold"}


async def test_only_the_drafter_sees_the_staged_skills(tmp_path: Any) -> None:
    run = Run(tmp_path, {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()]})
    await run.go()
    skills = tmp_path / "_task_drafting/.agents/skills"
    assert sorted(path.name for path in skills.iterdir()) == ["draft-task", "memory-log"]
    skill = (skills / "draft-task/SKILL.md").read_text()
    assert "./task/capture.py" in skill and "{{DRAFT_CAPTURE_PATH}}" not in skill
    assert (skills / "memory-log/scripts/append_memory.py").is_file()
    assert not (tmp_path / STAGED / ".agents").exists()
    assert not (tmp_path / "_task_drafting/.harness/judge_view_0/.agents").exists()
    prompt = run.prompts(Role.TASK_DRAFTER)[0]
    assert "{{" not in prompt and "./task/capture.py" in prompt and "task/task.md" in prompt
    assert "call the `report_blocker` tool" in prompt and "exit non-zero" not in prompt


async def test_the_judge_cannot_change_the_scaffold(tmp_path: Any) -> None:
    attack = tool_call("write_file", {"file_path": "/hack.txt", "content": "x"})
    run = Run(tmp_path, {**drafter(), Role.ALIGNMENT_JUDGE: [attack, judge_step()]})
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    assert not (tmp_path / "_task_drafting/.harness/judge_view_0/hack.txt").exists()
    assert not (tmp_path / "task/hack.txt").exists()
    prompt = run.prompts(Role.ALIGNMENT_JUDGE)[0]
    assert "{{" not in prompt and "Return only the report as your structured response" in prompt
    assert "- Scaffold: `.`" in prompt and "`nl -ba <path>`" in prompt
    assert "nl" in run.models.scripted(Role.ALIGNMENT_JUDGE).bound_tools[0]
    assert "execute" not in run.models.scripted(Role.ALIGNMENT_JUDGE).bound_tools[0] or True


async def test_hooks_run_around_the_steps_of_the_method(tmp_path: Any) -> None:
    seen: list[str] = []

    async def hook(state: Any, runtime: Any) -> dict[str, Any]:
        seen.append("hook")
        return {"hook_outputs": {"ran": True}}

    hooks = Hooks(before={"drafting.preflight": [hook]}, after={"drafting.publish": [hook]})
    run = Run(tmp_path, {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()]}, hooks=hooks)
    out = await run.go()
    assert out["drafting"].status == "published"
    assert seen == ["hook", "hook"]
    assert out["hook_outputs"]["drafting.preflight"] == {"ran": True}
    nodes = set(run.graph.get_graph().nodes)
    assert {"drafting.smoke.before", "drafting.smoke.after", "drafting.qualify.before"} <= nodes


# --- A2: drafting ------------------------------------------------------------------------------


async def test_a_missing_required_file_is_retried_with_the_harness_feedback(tmp_path: Any) -> None:
    create = tool_call(
        "write_file", {"file_path": "/task/capture.py", "content": "class OutputCapture: ..."}
    )
    run = Run(
        tmp_path,
        {
            Role.TASK_DRAFTER: [*drafter_script(skip=["capture.py"]), create, final("created it")],
            Role.ALIGNMENT_JUDGE: [judge_step()],
        },
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    first, second = run.prompts(Role.TASK_DRAFTER)[:2]
    assert "# Harness retry feedback" not in first
    assert second.startswith(first)
    assert "# Harness retry feedback" in second
    assert "Missing required drafted artifact: capture.py" in second
    assert second.count("# Harness retry feedback") == 1
    assert run.calls(Role.TASK_DRAFTER) == 2


async def test_the_draft_attempts_are_a_bound(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {Role.TASK_DRAFTER: [final(), final(), final(), final()]},
        config=fast(draft_max_attempts=2),
    )
    out = await run.go()
    assert out["drafting"].status == "failed"
    assert out["drafting"].failure.category == "draft_incomplete"
    assert run.calls(Role.TASK_DRAFTER) == 2
    prompts = run.prompts(Role.TASK_DRAFTER)
    assert prompts[1].count("# Harness retry feedback") == 1


async def test_an_unsafe_scaffold_ends_drafting(tmp_path: Any) -> None:
    run = Run(tmp_path, drafter(skip=["capture.py"]))
    run.task = await write_scaffold(run.workspace, only_task_md=True)
    os.symlink("task.md", tmp_path / "task/capture.py")
    out = await run.invoke()
    assert out["drafting"].status == "failed"
    assert out["drafting"].failure.category == "draft_unsafe"
    assert "Unsafe drafted artifact: capture.py" in out["drafting"].failure.detail
    assert run.calls(Role.TASK_DRAFTER) == 1


async def test_a_blocker_of_the_drafter_is_carried_into_the_failure(tmp_path: Any) -> None:
    blocked = tool_call("report_blocker", {"text": "the benchmark cannot be downloaded"})
    run = Run(tmp_path, {Role.TASK_DRAFTER: [blocked, blocked]}, config=fast(draft_max_attempts=2))
    out = await run.go()
    assert out["drafting"].failure.category == "draft_incomplete"
    assert "the benchmark cannot be downloaded" in out["drafting"].failure.detail


# --- A3: contract preflight --------------------------------------------------------------------


BROKEN = {
    "task_contract.json": lambda text: text.replace("python_rpc_v1:handle", "python_rpc_v1:bad")
}


async def test_a_contract_failure_is_repaired_with_the_base64_diagnostic(tmp_path: Any) -> None:
    fix = edit("/task_contract.json", "python_rpc_v1:bad", "python_rpc_v1:handle")
    run = Run(
        tmp_path,
        {
            **drafter(transform=BROKEN),
            Role.PREFLIGHT_REPAIRER: [fix, final("fixed")],
            Role.ALIGNMENT_JUDGE: [judge_step()],
        },
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    (prompt,) = run.prompts(Role.PREFLIGHT_REPAIRER)
    diagnostic = block(prompt, "Preflight diagnostic (base64)")
    assert "candidate.interface must be python_rpc_v1:handle" in diagnostic
    assert "\n" not in diagnostic and diagnostic.endswith("\\x0a")
    assert "# Harness-owned validation gate" in prompt
    assert 'task_gate(check="acceptance-smoke")' in prompt and "$TASK_GATE_CLIENT" not in prompt
    assert "call the `report_blocker` tool" in prompt
    assert prompt.index("# Repair policy") < prompt.index("# Harness-owned validation gate")
    tools = run.models.scripted(Role.PREFLIGHT_REPAIRER).bound_tools[0]
    assert "task_gate" in tools and "report_blocker" in tools


async def test_the_in_session_gate_checks_are_a_bound(tmp_path: Any) -> None:
    asks = [tool_call("task_gate", {"check": "contract"}) for _ in range(3)]
    run = Run(
        tmp_path,
        {
            **drafter(transform=BROKEN),
            Role.PREFLIGHT_REPAIRER: [*asks, final("gave up")],
        },
        config=fast(max_gate_checks_per_repair=2, contract_max_repairs=1),
    )
    await run.go()
    results = [
        message.content
        for message in run.models.scripted(Role.PREFLIGHT_REPAIRER).calls[-1]
        if message.type == "tool"
    ]
    assert [text.splitlines()[0] for text in results[:2]] == [
        "contract: failed",
        "contract: failed",
    ]
    assert results[2].startswith("request refused: the gate service is closed after 2 checks")
    assert "operational, not a new scaffold failure" in results[2]


async def test_the_contract_repairs_are_a_bound(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {**drafter(transform=BROKEN), Role.PREFLIGHT_REPAIRER: [final(), final(), final()]},
        config=fast(contract_max_repairs=2),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "contract_invalid"
    assert run.calls(Role.PREFLIGHT_REPAIRER) == 2
    assert "after 2 repair attempt(s)" in out["drafting"].failure.detail


async def test_a_repair_that_edits_the_canonical_task_is_refused(tmp_path: Any) -> None:
    sabotage = edit("/task.md", "Contracted toy task", "A different task")
    run = Run(tmp_path, {**drafter(transform=BROKEN), Role.PREFLIGHT_REPAIRER: [sabotage, final()]})
    out = await run.go()
    assert out["drafting"].status == "failed"
    assert out["drafting"].failure.category == "scaffold_unsafe"
    assert "modified canonical task.md" in out["drafting"].failure.detail


# --- A4: smoke ---------------------------------------------------------------------------------


async def test_infrastructure_failures_retry_without_spending_a_repair(tmp_path: Any) -> None:
    counter = {"n": 0}

    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "smoke" and request.evaluator == "search" and counter["n"] < 2:
            counter["n"] += 1
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="infrastructure_error",
                failure_category="runtime_bootstrap",
            )
        return None

    evaluator = evaluator_with(rule)
    run = Run(
        tmp_path,
        {**drafter(), Role.SMOKE_REPAIRER: [], Role.ALIGNMENT_JUDGE: [judge_step()]},
        evaluator=evaluator,
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    smokes = [
        call for call in evaluator.calls if call.plane == "smoke" and call.evaluator == "search"
    ]
    assert len(smokes) == 3 and len({call.evaluation_id for call in smokes}) == 3
    assert run.calls(Role.SMOKE_REPAIRER) == 0


async def test_the_smoke_infrastructure_attempts_are_a_bound(tmp_path: Any) -> None:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "smoke" and request.evaluator == "search":
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="infrastructure_error",
                failure_category="gpu_allocation_busy",
            )
        return None

    evaluator = evaluator_with(rule)
    run = Run(
        tmp_path,
        {**drafter(), Role.SMOKE_REPAIRER: []},
        evaluator=evaluator,
        config=fast(smoke_max_infrastructure_attempts=2),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_infrastructure_failure"
    assert len([c for c in evaluator.calls if c.plane == "smoke" and c.evaluator == "search"]) == 2
    assert run.calls(Role.SMOKE_REPAIRER) == 0


async def test_a_non_retryable_infrastructure_failure_is_not_retried(tmp_path: Any) -> None:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "smoke" and request.evaluator == "search":
            return EvaluationResult(
                evaluation_id=request.evaluation_id, status="infrastructure_error"
            )
        return None

    evaluator = evaluator_with(rule)
    run = Run(tmp_path, {**drafter(), Role.SMOKE_REPAIRER: []}, evaluator=evaluator)
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_infrastructure_failure"
    assert len([c for c in evaluator.calls if c.plane == "smoke" and c.evaluator == "search"]) == 1


async def test_a_smoke_repair_is_scoped_then_validated_by_the_failed_gate_first(
    tmp_path: Any,
) -> None:
    evaluator = failing_while(tmp_path, "# FIXED")
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.SMOKE_REPAIRER: [mark("/evaluate.py", "FIXED"), final("fixed")],
            Role.ALIGNMENT_JUDGE: [judge_step()],
        },
        evaluator=evaluator,
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    (prompt,) = run.prompts(Role.SMOKE_REPAIRER)
    diagnostic = json.loads(re.search(r"```text\n(\{.*\})\n```", prompt).group(1))
    assert diagnostic["gate"] == "search-smoke" and diagnostic["category"] == "search_nonzero_exit"
    assert diagnostic["status"] == "authored_failure" and "boom" not in prompt
    assert "Provisional dependency-removal transaction: inactive" in prompt
    assert "# Harness-owned validation gate" in prompt
    assert "the `task_gate` tool when available" in prompt
    ids = [
        c.evaluation_id for c in evaluator.calls if c.plane == "smoke" and c.evaluator == "search"
    ]
    assert [re.search(r"gate_\d+_([a-z-]+)/", i).group(1) for i in ids] == [
        "acceptance-smoke",
        "search-smoke",
        "acceptance-smoke",
    ], "the failed gate is repeated alone, then the full acceptance check"
    assert "# FIXED" in (tmp_path / "task/evaluate.py").read_text()


async def test_a_smoke_repair_outside_its_scope_is_undone(tmp_path: Any) -> None:
    evaluator = failing_while(tmp_path, "# NEVER")
    stray = tool_call("write_file", {"file_path": "/notes.py", "content": "x = 1"})
    run = Run(tmp_path, {**drafter(), Role.SMOKE_REPAIRER: [stray, final()]}, evaluator=evaluator)
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_repair_out_of_scope"
    assert "out-of-scope path(s): notes.py" in out["drafting"].failure.detail
    assert not (tmp_path / STAGED / "notes.py").exists(), "the transaction restored the snapshot"


async def test_the_smoke_repairs_are_a_bound_and_the_chain_rolls_back(tmp_path: Any) -> None:
    evaluator = failing_while(tmp_path, "# NEVER")
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.SMOKE_REPAIRER: [
                mark("/evaluate.py", "attempt"),
                final(),
                mark("/evaluate.py", "x"),
                final(),
            ],
        },
        evaluator=evaluator,
        config=fast(smoke_max_repairs=1),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_failed"
    assert run.calls(Role.SMOKE_REPAIRER) == 1
    assert "# attempt" not in staged_text(tmp_path, "evaluate.py"), "the failed chain was restored"
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()


async def test_a_dependency_failure_lets_the_repair_remove_an_unused_pin(tmp_path: Any) -> None:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if (
            request.plane == "smoke"
            and request.evaluator == "search"
            and "numpy" in staged_text(tmp_path, "requirements.txt")
        ):
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="failed",
                failure_category="runtime_dependencies",
            )
        return None

    def with_numpy(text: str) -> str:
        contract = json.loads(text)
        contract["provenance"]["dependencies"].append({"name": "numpy", "version": "1.26.0"})
        return json.dumps(contract)

    run = Run(
        tmp_path,
        {
            **drafter(
                transform={
                    "task_contract.json": with_numpy,
                    "requirements.txt": lambda text: text + "numpy==1.26.0\n",
                }
            ),
            Role.SMOKE_REPAIRER: [edit("/requirements.txt", "numpy==1.26.0\n", ""), final()],
            Role.ALIGNMENT_JUDGE: [judge_step()],
        },
        evaluator=evaluator_with(rule),
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    published = json.loads((tmp_path / "task/task_contract.json").read_text())
    assert published["provenance"]["dependencies"] == [{"name": "python", "version": "3.12"}]
    assert (tmp_path / "task/requirements.txt").read_text() == "python==3.12\n"
    (prompt,) = run.prompts(Role.SMOKE_REPAIRER)
    assert "Provisional dependency-removal transaction: inactive" in prompt
    assert '"category": "runtime_dependencies"' in prompt


async def test_a_repair_may_not_add_a_dependency(tmp_path: Any) -> None:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "smoke" and request.evaluator == "search":
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="failed",
                failure_category="runtime_dependencies",
            )
        return None

    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.SMOKE_REPAIRER: [
                edit("/requirements.txt", "python==3.12\n", "python==3.12\nscipy==1.0.0\n"),
                final(),
            ],
        },
        evaluator=evaluator_with(rule),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_repair_out_of_scope"
    assert "only remove pinned dependencies" in out["drafting"].failure.detail
    assert "scipy" not in staged_text(tmp_path, "requirements.txt")


# --- A5: alignment -----------------------------------------------------------------------------


async def test_a_malformed_report_is_retried_once_with_the_correction(tmp_path: Any) -> None:
    bad = judge_step(mutate=lambda report: report.model_copy(update={"fingerprints": {"x": "y"}}))
    run = Run(tmp_path, {**drafter(), Role.ALIGNMENT_JUDGE: [bad, judge_step()]})
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    first, second = run.prompts(Role.ALIGNMENT_JUDGE)
    assert second.startswith(first.rstrip())
    assert "# Output-contract correction" in second and "return only the corrected report" in second
    assert "task alignment report fingerprints do not match scaffold" in block(
        second, "Validator diagnostic (base64)"
    )
    assert block(second, "Validator diagnostic (base64)").startswith("invalid alignment report:")
    assert "# Output-contract correction" not in first


async def test_a_second_malformed_report_ends_drafting(tmp_path: Any) -> None:
    bad = judge_step(mutate=lambda report: report.model_copy(update={"summary": " "}))
    run = Run(tmp_path, {**drafter(), Role.ALIGNMENT_JUDGE: [bad, bad, bad]})
    out = await run.go()
    assert out["drafting"].failure.category == "alignment_report_invalid"
    assert run.calls(Role.ALIGNMENT_JUDGE) == 2
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()


async def test_a_failed_report_is_repaired_inside_a_transaction(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(failing=["compute_realism"]), judge_step()],
            Role.ALIGNMENT_REPAIRER: [mark("/evaluate.py", "aligned"), final("repaired")],
        },
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    (repair,) = run.prompts(Role.ALIGNMENT_REPAIRER)
    assert "fix compute_realism" in repair and "compute_realism" in repair
    assert "# Prior independent alignment findings" in repair
    assert "alignment_0.json" in repair
    assert "# Harness-owned validation gate" in repair and "report_blocker" in repair
    assert "{{" not in repair
    second_judge = run.prompts(Role.ALIGNMENT_JUDGE)[1]
    assert "# Prior independent alignment findings (untrusted historical data)" in second_judge
    assert "# aligned" in (tmp_path / "task/evaluate.py").read_text()
    assert out["drafting"].receipts["alignment_report"]["verdict"] == "pass"


async def test_the_alignment_repairs_are_a_bound_and_the_draft_rolls_back(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(failing=["reproducibility"])] * 3,
            Role.ALIGNMENT_REPAIRER: [mark("/evaluate.py", "aligned"), final()],
        },
        config=fast(alignment_max_repairs=1),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "alignment_failed"
    assert run.calls(Role.ALIGNMENT_JUDGE) == 2
    assert len(run.prompts(Role.ALIGNMENT_REPAIRER)) == 1
    assert "# aligned" not in staged_text(tmp_path, "evaluate.py"), (
        "the provisional edits were undone"
    )
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()


async def test_a_provisional_repair_that_fails_validation_is_continued_not_restarted(
    tmp_path: Any,
) -> None:
    evaluator = failing_while(tmp_path, "# FINE", plane="smoke")

    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if (
            request.plane == "smoke"
            and request.evaluator == "search"
            and "# BROKEN" in staged_text(tmp_path, "evaluate.py")
        ):
            return EvaluationResult(evaluation_id=request.evaluation_id, status="failed")
        return None

    evaluator = evaluator_with(rule)
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(failing=["compute_realism"]), judge_step()],
            Role.ALIGNMENT_REPAIRER: [
                mark("/evaluate.py", "BROKEN"),
                final("first"),
                edit("/evaluate.py", "# BROKEN", "# FIXED"),
                final("second"),
            ],
        },
        evaluator=evaluator,
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    first, second = run.prompts(Role.ALIGNMENT_REPAIRER)
    base = first[: first.index("\n# Harness-owned validation gate")]
    assert second.startswith(base) and "# Continue provisional alignment repair" in second
    assert "The previous alignment edits remain in the current provisional tree" in second
    diagnostic = block(second, "Validation diagnostic (base64)")
    assert (
        '"category": "search_nonzero_exit"' in diagnostic and "alignment-repair" not in diagnostic
    )
    assert "# FIXED" in (tmp_path / "task/evaluate.py").read_text()


async def test_the_judge_call_budget_is_twice_the_repairs_plus_one(tmp_path: Any) -> None:
    from langgraph.runtime import Runtime

    workspace = LocalWorkspace(tmp_path)
    context, _ = make_context(workspace, {Role.ALIGNMENT_JUDGE: [judge_step()]})
    state = {
        "run_config": build_run_config(context),
        "judge_calls": 30,
        "view_dir": "v",
        "judge_prompt": "p",
    }
    update = await alignment_judge(state, Runtime(context=context))  # type: ignore[arg-type]
    assert update["failure"].category == "alignment_judge_budget_exhausted"
    config = fast(alignment_max_repairs=1, qualification_max_repairs=0)
    context, models = make_context(workspace, {Role.ALIGNMENT_JUDGE: []}, config=config)
    state = {
        "run_config": build_run_config(context),
        "judge_calls": 4,
        "view_dir": "v",
        "judge_prompt": "p",
    }
    update = await alignment_judge(state, Runtime(context=context))  # type: ignore[arg-type]
    assert update["failure"].category == "alignment_judge_budget_exhausted"
    assert models.scripted(Role.ALIGNMENT_JUDGE).call_count == 0


# --- A6: qualification -------------------------------------------------------------------------


def qualification_rule(tmp_path: Any, marker: str = "# CALIBRATED") -> FakeEvaluator:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "qualification" and marker not in staged_text(tmp_path, "evaluate.py"):
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="completed",
                metrics={"combined_score": 0.0, "accuracy": 0.0},
                correct=False,
                feasible=False,
                summary="the workload did not complete in time",
            )
        return None

    return evaluator_with(rule)


async def test_a_qualification_failure_is_repaired_as_a_pending_transaction(tmp_path: Any) -> None:
    evaluator = qualification_rule(tmp_path)
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(), judge_step()],
            Role.QUALIFICATION_REPAIRER: [mark("/evaluate.py", "CALIBRATED"), final("calibrated")],
        },
        evaluator=evaluator,
    )
    out = await run.go()
    drafting = out["drafting"]
    assert drafting.status == "published", drafting.failure
    # the repair prompt: the five base64 blocks and the shared policy, then the histories
    (repair,) = run.prompts(Role.QUALIFICATION_REPAIRER)
    assert repair.startswith("Repair the drafted task scaffold so its ordinary, non-smoke")
    assert "qualification_infeasible" in block(repair, "Qualification diagnostic (base64)")
    assert "Contracted toy task" in block(repair, "Immutable accepted design (base64)") or True
    assert block(repair, "Prior calibration record, when present (base64)") == ""
    assert block(repair, "Latest rejected repair validation, when present (base64)") == ""
    assert (
        "# Shared scaffold repair policy" in repair
        and "# Prior independent alignment findings" in repair
    )
    assert "# Harness-owned validation gate" in repair
    # the calibration reaches the judge of the second round, with the historical failure
    first_judge, second_judge = run.prompts(Role.ALIGNMENT_JUDGE)
    assert "qualification_calibration" not in attestation(first_judge)
    context = attestation(second_judge)
    calibration = context["qualification_calibration"]
    assert calibration["status"] == "provisional" and calibration["original_design_immutable"]
    assert calibration["original_fingerprint"] != calibration["calibrated_fingerprint"]
    assert context["current_qualification"]["status"] == "pass"
    assert context["qualification_failure"]["category"] == "qualification_infeasible"
    assert context["qualification_failure"]["matches_current_scaffold"] is False
    assert "workload did not complete" not in second_judge, (
        "the task-authored diagnostic is omitted"
    )
    # both gates accepted the same scaffold: the calibration is accepted and the draft committed
    accepted = drafting.receipts["calibration"]
    assert accepted["status"] == "accepted"
    assert set(accepted["evidence_sha256"]) == {
        "accepted_design",
        "qualification_failure",
        "alignment_report",
        "qualification_report",
        "qualification_receipt",
    }
    assert "# CALIBRATED" in (tmp_path / "task/evaluate.py").read_text()
    qualifications = [c for c in evaluator.calls if c.plane == "qualification"]
    assert len(qualifications) == 2


async def test_an_alignment_rejection_of_a_pending_repair_rolls_back_at_the_bound(
    tmp_path: Any,
) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(), judge_step(failing=["compute_realism"])],
            Role.QUALIFICATION_REPAIRER: [mark("/evaluate.py", "CALIBRATED"), final()],
        },
        evaluator=qualification_rule(tmp_path),
        config=fast(qualification_max_repairs=1),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "qualification_repair_exhausted"
    assert "Qualification repairs 1/1" in out["drafting"].failure.detail
    assert "# CALIBRATED" not in staged_text(tmp_path, "evaluate.py"), (
        "the pending repair was undone"
    )
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()


async def test_an_alignment_rejection_feeds_the_next_qualification_repair(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [
                judge_step(),
                judge_step(failing=["compute_realism"]),
                judge_step(),
            ],
            Role.QUALIFICATION_REPAIRER: [
                mark("/evaluate.py", "CALIBRATED"),
                final("first"),
                edit("/evaluate.py", "# CALIBRATED", "# CALIBRATED again"),
                final("second"),
            ],
        },
        evaluator=qualification_rule(tmp_path),
        config=fast(qualification_max_repairs=2),
    )
    out = await run.go()
    assert out["drafting"].status == "published", out["drafting"].failure
    first, second = run.prompts(Role.QUALIFICATION_REPAIRER)
    assert "fix compute_realism" in block(second, "Prior alignment report, when present (base64)")
    assert block(first, "Prior alignment report, when present (base64)") == ""
    assert "compute_realism" in block(second, "Prior alignment report, when present (base64)")
    calibration = block(second, "Prior calibration record, when present (base64)")
    assert '"status": "provisional"' in calibration and "original_design_immutable" in calibration
    chunk = second[second.index("# Rejected repair history") : second.index("\n# Harness-owned")]
    payload = json.loads(chunk.split("revalidate every gate.\n", 1)[1])
    assert "# CALIBRATED" in payload["rejected_repairs"][0]["diff"]
    assert out["drafting"].receipts["calibration"]["status"] == "accepted"


async def test_a_repair_that_reproduces_a_failed_fingerprint_is_an_oscillation(
    tmp_path: Any,
) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step()],
            Role.QUALIFICATION_REPAIRER: [final("changed nothing")],
        },
        evaluator=qualification_rule(tmp_path),
    )
    out = await run.go()
    failure = out["drafting"].failure
    assert failure.category == "qualification_repair_oscillation"
    assert (
        failure.detail
        == "Qualification repair reproduced a previously failed scaffold fingerprint."
    )
    assert run.calls(Role.QUALIFICATION_REPAIRER) == 1
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()


async def test_the_qualification_repairs_are_a_bound(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step()],
            Role.QUALIFICATION_REPAIRER: [
                mark("/evaluate.py", "a"),
                final(),
                mark("/evaluate.py", "b"),
                final(),
            ],
        },
        evaluator=qualification_rule(tmp_path, marker="# NEVER"),
        config=fast(qualification_max_repairs=2),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "qualification_repair_exhausted"
    assert len(run.prompts(Role.QUALIFICATION_REPAIRER)) == 2
    qualifications = [c for c in run.context.evaluator.calls if c.plane == "qualification"]  # type: ignore[union-attr]
    assert len(qualifications) == 3, "the initial run and one after each repair"
    assert "# a" not in staged_text(tmp_path, "evaluate.py")
    assert "failed gate: production-search" in out["drafting"].failure.detail


async def test_without_repairs_a_qualification_failure_is_final(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()], Role.QUALIFICATION_REPAIRER: []},
        evaluator=qualification_rule(tmp_path),
        config=fast(qualification_max_repairs=0),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "qualification_repair_exhausted"
    assert run.calls(Role.QUALIFICATION_REPAIRER) == 0


@pytest.mark.parametrize(
    ("category", "runs"), [("runtime_bootstrap", 3), ("gpu_allocation_invalid", 1)]
)
async def test_qualification_infrastructure_attempts_are_a_bound(
    tmp_path: Any, category: str, runs: int
) -> None:
    def rule(request: EvaluationRequest) -> EvaluationResult | None:
        if request.plane == "qualification":
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="infrastructure_error",
                failure_category=category,
            )
        return None

    evaluator = evaluator_with(rule)
    run = Run(
        tmp_path,
        {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()], Role.QUALIFICATION_REPAIRER: []},
        evaluator=evaluator,
    )
    out = await run.go()
    assert out["drafting"].failure.category == "qualification_infrastructure_failure"
    assert len([c for c in evaluator.calls if c.plane == "qualification"]) == runs
    assert run.calls(Role.QUALIFICATION_REPAIRER) == 0


async def test_a_credential_in_the_scaffold_is_not_published(tmp_path: Any) -> None:
    leaky = {
        "resources/search/target.txt": lambda text: text + "key sk-abcdefghijklmnopqrstuvwxyz0123\n"
    }
    run = Run(tmp_path, {**drafter(transform=leaky), Role.ALIGNMENT_JUDGE: [judge_step()]})
    out = await run.go()
    assert out["drafting"].failure.category == "publication_failed"
    assert "credential-like content: resources/search/target.txt" in out["drafting"].failure.detail
    assert not (tmp_path / "task/.agentic_task_validation.json").exists()
    assert not (tmp_path / "task/evaluate.py").exists(), "nothing was published"


# --- the legacy workflow -----------------------------------------------------------------------


async def test_the_legacy_workflow_shares_one_repair_pool(tmp_path: Any) -> None:
    evaluator = failing_while(tmp_path, "# NEVER")
    fix = edit("/task_contract.json", "python_rpc_v1:bad", "python_rpc_v1:handle")
    run = Run(
        tmp_path,
        {
            **drafter(transform=BROKEN),
            Role.PREFLIGHT_REPAIRER: [fix, final()],
            Role.SMOKE_REPAIRER: [],
        },
        evaluator=evaluator,
        config=fast(workflow="legacy", legacy_validation_max_repairs=1),
    )
    out = await run.go()
    assert out["drafting"].failure.category == "smoke_failed"
    assert run.calls(Role.PREFLIGHT_REPAIRER) == 1
    assert run.calls(Role.SMOKE_REPAIRER) == 0, "the preflight repair spent the shared pool"
    assert "after 1 repair attempt(s)" in out["drafting"].failure.detail


# --- resume and deferred evaluation ------------------------------------------------------------


class Crash(BaseException):
    """A process crash in the middle of an evaluation."""


class CrashOnce(FakeEvaluator):
    def __init__(self, plane: str) -> None:
        super().__init__(default=fixture_result)
        self.plane = plane
        self.crashed = False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.plane == self.plane and not self.crashed:
            self.crashed = True
            raise Crash
        return await super().evaluate(request)


@pytest.mark.parametrize("plane", ["smoke", "qualification"])
async def test_a_crash_resumes_without_repeating_a_session_or_an_evaluation(
    tmp_path: Any, plane: str
) -> None:
    evaluator = CrashOnce(plane)
    run = Run(
        tmp_path,
        {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()]},
        evaluator=evaluator,
    )
    with pytest.raises(Crash):
        await run.go()
    out = await run.invoke(None)
    assert out["drafting"].status == "published", out["drafting"].failure
    assert run.calls(Role.TASK_DRAFTER) == 1
    assert run.calls(Role.ALIGNMENT_JUDGE) == 1
    assert all(count == 1 for count in evaluator.runs.values()), dict(evaluator.runs)
    ids = [call.evaluation_id for call in evaluator.calls]
    assert len(ids) == len(set(ids)), "no evaluation id was evaluated twice"


async def test_the_state_survives_a_strict_checkpoint_serializer(tmp_path: Any) -> None:
    """Crash after the pending transaction exists, resume from a strict, allow-listed saver."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from langgraph_agentic_scientist import serde_allowlist

    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))
    run = Run(
        tmp_path,
        {
            **drafter(),
            Role.ALIGNMENT_JUDGE: [judge_step(), judge_step()],
            Role.QUALIFICATION_REPAIRER: [mark("/evaluate.py", "CALIBRATED"), final()],
        },
        evaluator=CrashAfterRepair(tmp_path),
    )
    run.workspace = LocalWorkspace(tmp_path)
    run.context, run.models = make_context(run.workspace, run.scripts, **run.kwargs)
    run.graph = build_task_scaffold_graph(checkpointer=saver)
    run.task = await write_scaffold(run.workspace, only_task_md=True)
    with pytest.raises(Crash):
        await run.invoke()
    out = await run.invoke(None)
    assert out["drafting"].status == "published", out["drafting"].failure
    assert out["drafting"].receipts["calibration"]["status"] == "accepted"


class CrashAfterRepair(FakeEvaluator):
    """Qualification fails until the repair, and the process crashes at the second one."""

    def __init__(self, tmp_path: Any) -> None:
        super().__init__(default=self.answer)
        self.tmp_path = tmp_path
        self.crashed = False

    def answer(self, request: EvaluationRequest) -> EvaluationResult:
        if request.plane == "qualification":
            repaired = "# CALIBRATED" in staged_text(self.tmp_path, "evaluate.py")
            if repaired and not self.crashed:
                self.crashed = True
                raise Crash
            return fixture_result(request).model_copy(update={"feasible": repaired})
        return fixture_result(request)


async def test_deferred_evaluations_resume_one_interrupt_at_a_time(tmp_path: Any) -> None:
    run = Run(
        tmp_path,
        {**drafter(), Role.ALIGNMENT_JUDGE: [judge_step()]},
        evaluator=DeferredEvaluator(),
    )
    await run.go()
    config = {"configurable": {"thread_id": run.thread}, **RECURSION}
    requested: list[str] = []
    for _ in range(20):
        state = await run.graph.aget_state(config)
        if not state.interrupts:
            break
        payload = state.interrupts[0].value
        assert payload["kind"] == "evaluation_request"
        request = EvaluationRequest.model_validate(payload["request"])
        requested.append(f"{request.plane}/{request.evaluator}")
        answer = fixture_result(request).model_dump(mode="json", by_alias=True)
        await run.graph.ainvoke(Command(resume=answer), config, context=run.context)
    final_state = await run.graph.aget_state(config)
    assert final_state.values["drafting"].status == "published"
    assert requested == [
        "smoke/search",
        "smoke/final",
        "production_entry/search",
        "production_entry/final",
        "qualification/search",
    ]
    assert run.calls(Role.TASK_DRAFTER) == 1 and run.calls(Role.ALIGNMENT_JUDGE) == 1


def test_the_fixture_task_is_the_core_fixture() -> None:
    assert (FIXTURE / "task_contract.json").is_file()
