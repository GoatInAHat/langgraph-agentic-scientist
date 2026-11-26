"""Turning a role into a runnable agent (design decision D1).

Upstream runs every role as an external coding-agent CLI in a working directory with staged
skills. The port runs a role as a Deep Agents agent (``deepagents.create_deep_agent``): its
built-in tools (``ls``, ``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``,
``delete``, ``execute``, ``task``) are the coding agent's tool set, its ``SkillsMiddleware`` loads
the staged ``SKILL.md`` folders, and its backend is the role's view of the workspace.

:class:`AgentFactory` is a protocol, so a caller can substitute its own (for example one that runs
each session in a remote sandbox); the runner invokes whatever it returns as a subgraph of the
calling node. :class:`DeepAgentFactory` is the default and sets, for every agent:

* the structured-output strategy chosen per role by
  :class:`~langgraph_agentic_scientist.ModelRegistry` (``ToolStrategy`` for ``function_calling``,
  ``ProviderStrategy(strict=True)`` for ``json_schema``), never left to the model profile;
* ``ModelRetryMiddleware`` with upstream's schedule for transient provider failures (15, 45 and
  90 s by default) and, when configured, ``ModelCallLimitMiddleware``;
* a ``FilesystemMiddleware`` whose maximum per-command timeout is the configured shell maximum
  (upstream's 7200 s; Deep Agents' own default is 3600 s);
* a copy of the model whose profile carries the role's context window when the registry gives
  one, so that Deep Agents summarises at 85 % of the real window (the caller's model object is
  never changed);
* the same middleware on Deep Agents' ``general-purpose`` subagent (the ``task`` tool), which
  otherwise inherits only the ``FilesystemMiddleware``: a delegated task is retried and bounded
  as the session that delegates it.

:meth:`DeepAgentFactory.middleware` is the place to add middleware of one's own (a subclass that
extends its list); the subagent gets the same list.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..models import Role, StructuredMethod
from ..workspace.readonly import ReadOnlyBackend

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..context import ScientistContext

SKILLS_SOURCE = "/.agents/skills/"
"""Backend path of the staged skills, relative to the role's root."""

TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
    }
)


def is_transient_provider_error(exc: BaseException) -> bool:
    """Return whether a model-call failure is transient (upstream retries these, 15/45/90 s).

    Connection errors, timeouts, HTTP 408/409/425/429/5xx/529 and the transient error classes of
    the common provider SDKs and HTTP clients are transient; everything else is not.
    """
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS_CODES
    return any(cls.__name__ in _TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__)


def response_format_for(schema: type[BaseModel], method: StructuredMethod) -> Any:
    """Return the explicit structured-output strategy of a role."""
    if method == "json_schema":
        return ProviderStrategy(schema, strict=True)
    return ToolStrategy(schema)


def with_context_window(model: BaseChatModel, window: int | None) -> BaseChatModel:
    """Return ``model``, or a copy whose profile carries ``max_input_tokens=window``."""
    if window is None:
        return model
    profile = dict(model.profile or {})
    if profile.get("max_input_tokens") == window:
        return model
    profile["max_input_tokens"] = window
    return model.model_copy(update={"profile": profile})


def _has_window(model: BaseChatModel) -> bool:
    profile = model.profile
    return isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int)


@runtime_checkable
class AgentFactory(Protocol):
    """Builds the runnable unit of a role."""

    def workspace_agent(
        self,
        role: Role,
        *,
        ctx: ScientistContext,
        root: str,
        skills: Sequence[str],
        tools: Sequence[BaseTool],
        pool_index: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return an agent that edits files under ``root`` (writable backend)."""
        ...

    def structured_agent(
        self,
        role: Role,
        schema: type[BaseModel],
        *,
        ctx: ScientistContext,
        root: str | None,
        read_only: bool = True,
        skills: Sequence[str] = (),
        tools: Sequence[BaseTool] = (),
        pool_index: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return an agent whose product is one ``schema`` object (``structured_response``)."""
        ...

    def structured_call(
        self, role: Role, schema: type[BaseModel], *, ctx: ScientistContext
    ) -> Runnable[Any, Any]:
        """Return a plain model call with structured output (no tools, no workspace)."""
        ...


class DeepAgentFactory:
    """The default :class:`AgentFactory`, over Deep Agents (see the module docstring)."""

    def __init__(self) -> None:
        self._warned: set[Role] = set()

    def model_for(
        self, role: Role, ctx: ScientistContext, *, pool_index: int | None = None
    ) -> BaseChatModel:
        """Return the role's model with its context window applied to a copy."""
        model = ctx.models.for_role(role, pool_index=pool_index)
        window = ctx.models.window_for(role)
        if window is None and not _has_window(model) and role not in self._warned:
            self._warned.add(role)
            warnings.warn(
                f"the model of role {role.value!r} has no max_input_tokens profile and no "
                "context window is configured; Deep Agents will summarise at a fixed 170,000 "
                "tokens. Set ModelRegistry.context_window for served models.",
                stacklevel=3,
            )
        return with_context_window(model, window)

    def middleware(self, ctx: ScientistContext, backend: BackendProtocol) -> list[Any]:
        """Return the middleware the factory adds to every agent."""
        agents = ctx.config.agents
        retry = agents.model_retry
        stack: list[Any] = [
            FilesystemMiddleware(
                backend=backend, max_execute_timeout=agents.execute_max_timeout_seconds
            ),
            ModelRetryMiddleware(
                max_retries=retry.max_retries,
                retry_on=is_transient_provider_error,
                on_failure="error",
                backoff_factor=retry.backoff_factor,
                initial_delay=retry.initial_delay,
                max_delay=retry.max_delay,
                jitter=retry.jitter,
            ),
        ]
        if agents.model_call_limit is not None:
            stack.append(
                ModelCallLimitMiddleware(run_limit=agents.model_call_limit, exit_behavior="end")
            )
        return stack

    def backend(
        self,
        ctx: ScientistContext,
        root: str | None,
        *,
        read_only: bool,
        env: Mapping[str, str] | None = None,
    ) -> BackendProtocol:
        """Return the role's backend: the workspace's view of ``root`` (read-only or not)."""
        if root is None:
            return ReadOnlyBackend(StateBackend()) if read_only else StateBackend()
        agents = ctx.config.agents
        return ctx.workspace.agent_backend(
            root,
            read_only=read_only,
            execute_timeout=agents.execute_timeout_seconds,
            execute_max_timeout=agents.execute_max_timeout_seconds,
            env=env,
            max_output_bytes=agents.max_tool_output_bytes,
        )

    def subagents(
        self, ctx: ScientistContext, backend: BackendProtocol, skills: Sequence[str]
    ) -> list[SubAgent]:
        """Return Deep Agents' ``general-purpose`` subagent with the factory's middleware."""
        spec: SubAgent = {
            **GENERAL_PURPOSE_SUBAGENT,
            "middleware": self.middleware(ctx, backend),
        }
        if skills:
            spec["skills"] = [SKILLS_SOURCE]
        return [spec]

    def _build(
        self,
        role: Role,
        *,
        ctx: ScientistContext,
        backend: BackendProtocol,
        skills: Sequence[str],
        tools: Sequence[BaseTool],
        response_format: Any = None,
        pool_index: int | None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        return create_deep_agent(
            model=self.model_for(role, ctx, pool_index=pool_index),
            tools=list(tools),
            system_prompt=ctx.prompts.text("harness/agent_system").strip(),
            middleware=self.middleware(ctx, backend),
            subagents=self.subagents(ctx, backend, skills),
            skills=[SKILLS_SOURCE] if skills else None,
            backend=backend,
            response_format=response_format,
            name=role.value,
        )

    def workspace_agent(
        self,
        role: Role,
        *,
        ctx: ScientistContext,
        root: str,
        skills: Sequence[str],
        tools: Sequence[BaseTool],
        pool_index: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return a Deep Agents agent with a writable backend rooted at ``root``."""
        backend = self.backend(ctx, root, read_only=False, env=env)
        return self._build(
            role, ctx=ctx, backend=backend, skills=skills, tools=tools, pool_index=pool_index
        )

    def structured_agent(
        self,
        role: Role,
        schema: type[BaseModel],
        *,
        ctx: ScientistContext,
        root: str | None,
        read_only: bool = True,
        skills: Sequence[str] = (),
        tools: Sequence[BaseTool] = (),
        pool_index: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return a Deep Agents agent that answers with a ``schema`` object."""
        backend = self.backend(ctx, root, read_only=read_only, env=env)
        return self._build(
            role,
            ctx=ctx,
            backend=backend,
            skills=skills,
            tools=tools,
            response_format=response_format_for(schema, ctx.models.method_for(role)),
            pool_index=pool_index,
        )

    def structured_call(
        self, role: Role, schema: type[BaseModel], *, ctx: ScientistContext
    ) -> Runnable[Any, Any]:
        """Return ``model.with_structured_output(schema, method=<the role's method>)``."""
        return structured_output(self.model_for(role, ctx), schema, ctx.models.method_for(role))


def structured_output(
    model: BaseChatModel, schema: type[BaseModel], method: StructuredMethod
) -> Runnable[Any, Any]:
    """Return ``model.with_structured_output`` with an explicit method.

    LangChain's base implementation accepts no ``method`` argument and always uses tool calling;
    for such a model ``function_calling`` is requested without the argument.
    """
    if method == "json_schema":
        return model.with_structured_output(schema, method="json_schema", strict=True)
    try:
        return model.with_structured_output(schema, method="function_calling")
    except (TypeError, ValueError) as exc:
        if "method" not in str(exc) and "unsupported" not in str(exc).lower():
            raise
        return model.with_structured_output(schema)


__all__ = [
    "SKILLS_SOURCE",
    "AgentFactory",
    "DeepAgentFactory",
    "is_transient_provider_error",
    "response_format_for",
    "with_context_window",
]
