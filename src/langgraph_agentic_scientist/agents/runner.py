"""Running one agent session inside a graph node.

Every phase runs its roles through :func:`run_workspace_role` (roles that edit files) and
:func:`run_structured_role` (roles whose product is one structured object). Both render the
prompt from the registry, stage the role's skills, build the agent through the context's
:class:`~langgraph_agentic_scientist.agents.AgentFactory`, and invoke it **as a subgraph of the
calling node**: its steps are checkpointed under the node's namespace, so a crash or an interrupt
resumes the session from its last completed step instead of repeating tool calls.

Rules (design decisions D1 and D6):

* **One session per node execution.** Call one of these helpers at most once in a node; a retry
  of a role (review attempts, format attempts, repair rounds) is a graph loop with a counter in
  state, so that each session has its own checkpoint namespace.
* **The wall clock is caught in the node.** The session runs inside ``asyncio.timeout(<the role's
  session timeout>)``; when it expires the helper returns ``timed_out=True`` and the graph goes on
  as upstream does after a timed-out agent CLI (the outputs are still checked). The clock bounds
  each uninterrupted run of a session; a session resumed after a crash gets a fresh clock.
* **Errors.** An interrupt from inside the session propagates (the run stops and resumes later).
  A transient provider failure is retried first inside the session by ``ModelRetryMiddleware``,
  which repeats only the failed model call and keeps the session's work. One that it could not
  absorb propagates, so that the node's ``RetryPolicy`` runs the node again: that starts the
  session again from its prompt, as upstream's retry of a failed agent run does (a resume after
  a crash, by contrast, continues the session from its checkpoint). Any other failure becomes
  ``failure`` (category ``agent_run_failed``), and the phase's budgets decide what follows, as
  upstream's non-zero exit of the agent did.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from pydantic import BaseModel

from ..context import ScientistContext
from ..core.render import truncate_utf8
from ..core.state import Failure
from ..evaluation.protocol import EvaluationResult
from ..models import Role
from .factory import is_transient_provider_error
from .staging import stage_skills
from .tools import blocker_from_messages, evaluations_from_messages

FAILURE_DETAIL_BYTES = 4096


@dataclass
class RoleOutcome:
    """The outcome of one agent session.

    Attributes:
        role: The role that ran.
        prompt: The rendered prompt the session was given.
        timed_out: True when the wall clock ended the session.
        failure: Why the session failed, when it did.
        structured: The structured answer of a structured role.
        final_text: The text of the agent's last message.
        blocker: A blocker reported through ``report_blocker``.
        evaluations: Results of the session's ``evaluate_program`` calls, in call order.
        messages: The session's messages (as far as they were observed).
        staged_skills: The skills staged for the session.
    """

    role: Role
    prompt: str
    timed_out: bool = False
    failure: Failure | None = None
    structured: BaseModel | None = None
    final_text: str = ""
    blocker: str | None = None
    evaluations: list[EvaluationResult] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)
    staged_skills: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the session ended normally (no timeout, no failure)."""
        return not self.timed_out and self.failure is None


def agent_node_retry_policy() -> RetryPolicy:
    """Return the outer retry line of nodes that run an agent session.

    Upstream retries a transient provider failure of an agent run up to 4 times, waiting 15, 45
    and 90 seconds, and restarts the agent run each time. Inside a session ``ModelRetryMiddleware``
    retries the failed model call alone (the agent CLIs upstream runs retry their API calls
    themselves); this policy, put on the node (``add_node(..., retry_policy=
    agent_node_retry_policy())``), runs the node again when a transient failure escapes the
    session, which starts the session again from its prompt, as upstream's outer retry does.
    """
    return RetryPolicy(
        max_attempts=4,
        initial_interval=15.0,
        backoff_factor=3.0,
        max_interval=90.0,
        jitter=False,
        retry_on=is_transient_provider_error,
    )


def render_prompt(
    ctx: ScientistContext,
    *,
    prompt: str | None,
    prompt_key: str | None,
    values: Mapping[str, str] | None,
    fragments: bool,
    suffix: str,
) -> str:
    """Return the prompt text of a session (an explicit prompt, or a rendered registry entry)."""
    if prompt is None:
        if prompt_key is None:
            raise ValueError("give either prompt or prompt_key")
        prompt = ctx.prompts.render(prompt_key, values or {}, fragments=fragments)
    return prompt + suffix


def _last_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
    return ""


async def _invoke(
    agent: Any, prompt: str, timeout: float | None, outcome: RoleOutcome
) -> dict[str, Any] | None:
    """Stream the agent as a subgraph, keeping the last observed state."""
    last: dict[str, Any] | None = None
    inputs = {"messages": [HumanMessage(content=prompt)]}
    try:
        async with asyncio.timeout(timeout):
            async for chunk in agent.astream(inputs, stream_mode="values"):
                if isinstance(chunk, dict):
                    last = chunk
    except TimeoutError:
        outcome.timed_out = True
    except GraphBubbleUp:
        raise
    except Exception as exc:
        if is_transient_provider_error(exc):
            raise
        outcome.failure = Failure(
            category="agent_run_failed",
            scope="infrastructure",
            detail=truncate_utf8(f"{type(exc).__name__}: {exc}", FAILURE_DETAIL_BYTES),
        )
    return last


def _collect(outcome: RoleOutcome, state: Mapping[str, Any] | None) -> None:
    messages = list((state or {}).get("messages") or [])
    outcome.messages = messages
    outcome.final_text = _last_text(messages)
    outcome.blocker = blocker_from_messages(messages)
    outcome.evaluations = evaluations_from_messages(messages)


async def run_workspace_role(
    *,
    role: Role,
    runtime: Runtime[ScientistContext],
    root: str,
    prompt: str | None = None,
    prompt_key: str | None = None,
    values: Mapping[str, str] | None = None,
    fragments: bool = True,
    suffix: str = "",
    skills: Sequence[str] = (),
    skill_values: Mapping[str, str] | None = None,
    skill_sources: Mapping[str, str] | None = None,
    tools: Sequence[BaseTool] = (),
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    pool_index: int | None = None,
    stage: bool = True,
) -> RoleOutcome:
    """Run one session of a role that edits files under ``root``.

    Args:
        role: The role.
        runtime: The calling node's runtime.
        root: Workspace directory the agent works in (its backend root and shell directory).
        prompt: An explicit prompt; otherwise ``prompt_key`` is rendered with ``values``.
        prompt_key: Registry key of the role's prompt.
        values: Placeholder values of the prompt.
        fragments: Whether to expand the shared fragments (false for the design prompts).
        suffix: Text appended to the rendered prompt (upstream's retry feedback sections).
        skills: Skills to stage under ``<root>/.agents/skills/``.
        skill_values: Placeholder values of the staged ``SKILL.md`` files (default ``values``).
        skill_sources: Role-specific versions of skills: staged name -> registry skill.
        tools: Package tools given to the role.
        timeout: The session's wall clock in seconds; ``None`` is unbounded.
        env: Extra environment variables for the agent's commands.
        pool_index: The cycle's model-pool choice, for roles that use the pool.
        stage: Whether to stage ``skills`` before the session. ``False`` when the caller has
            already staged them (and perhaps changed the staged copies, as the paper phase
            watermarks the writer's templates); they are still given to the agent.
    """
    ctx = runtime.context
    text = render_prompt(
        ctx, prompt=prompt, prompt_key=prompt_key, values=values, fragments=fragments, suffix=suffix
    )
    outcome = RoleOutcome(role=role, prompt=text)
    if skills and stage:
        outcome.staged_skills = await stage_skills(
            ctx.workspace,
            root,
            skills,
            prompts=ctx.prompts,
            values=skill_values if skill_values is not None else (values or {}),
            sources=skill_sources,
        )
    agent = ctx.agent_factory.workspace_agent(
        role, ctx=ctx, root=root, skills=skills, tools=tools, pool_index=pool_index, env=env
    )
    _collect(outcome, await _invoke(agent, text, timeout, outcome))
    return outcome


async def run_structured_role(
    *,
    role: Role,
    schema: type[BaseModel],
    runtime: Runtime[ScientistContext],
    root: str | None = None,
    read_only: bool = True,
    prompt: str | None = None,
    prompt_key: str | None = None,
    values: Mapping[str, str] | None = None,
    fragments: bool = True,
    suffix: str = "",
    skills: Sequence[str] = (),
    skill_values: Mapping[str, str] | None = None,
    tools: Sequence[BaseTool] = (),
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    pool_index: int | None = None,
    stage: bool = True,
) -> RoleOutcome:
    """Run one session of a role whose product is one ``schema`` object.

    The answer is ``outcome.structured``. A session that ends without a structured answer, times
    out or fails is a failed attempt of the role (``outcome.failure`` is set); the caller decides
    whether to retry. Read-only roles get a
    :class:`~langgraph_agentic_scientist.workspace.ReadOnlyBackend` (``read_only=True``).
    ``stage`` is as for :func:`run_workspace_role`.
    """
    ctx = runtime.context
    text = render_prompt(
        ctx, prompt=prompt, prompt_key=prompt_key, values=values, fragments=fragments, suffix=suffix
    )
    outcome = RoleOutcome(role=role, prompt=text)
    if skills and stage:
        if root is None:
            raise ValueError("skills need a workspace root")
        outcome.staged_skills = await stage_skills(
            ctx.workspace,
            root,
            skills,
            prompts=ctx.prompts,
            values=skill_values if skill_values is not None else (values or {}),
        )
    agent = ctx.agent_factory.structured_agent(
        role,
        schema,
        ctx=ctx,
        root=root,
        read_only=read_only,
        skills=skills,
        tools=tools,
        pool_index=pool_index,
        env=env,
    )
    state = await _invoke(agent, text, timeout, outcome)
    _collect(outcome, state)
    response = (state or {}).get("structured_response")
    if isinstance(response, BaseModel):
        outcome.structured = response
    elif isinstance(response, Mapping):
        outcome.structured = schema.model_validate(response)
    if outcome.structured is None and outcome.failure is None:
        outcome.failure = Failure(
            category="structured_response_missing" if not outcome.timed_out else "timeout",
            scope="method",
            detail="the session ended without a structured answer",
        )
    return outcome


__all__ = [
    "RoleOutcome",
    "agent_node_retry_policy",
    "render_prompt",
    "run_structured_role",
    "run_workspace_role",
]
