"""A chat model that replays a script, for tests that run without a network or a real model.

:class:`ScriptedChatModel` replays a list of steps: an ``AIMessage`` (tool calls included), a
callable that receives the message list and returns an ``AIMessage``, or an exception to raise.
``bind_tools`` returns a bound copy that records the tools; structured output works with both of
the package's strategies: ``function_calling`` (the model answers with a tool call named after
the schema, see :func:`structured_call`) and ``json_schema`` (the model answers with JSON text,
see :func:`json_answer`).

:class:`ScriptedModels` is a :class:`~langgraph_agentic_scientist.ModelRegistry` that routes each
role to its own script and fails with the role's name when a script runs out.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..models import ModelRegistry, Role

ScriptStep = AIMessage | Callable[[list[BaseMessage]], AIMessage] | BaseException
"""One step of a script."""


class ScriptExhausted(AssertionError):
    """A scripted model was called more often than its script has steps."""


@dataclass
class _ScriptState:
    """Mutable state shared by a scripted model and every copy and binding of it."""

    steps: list[ScriptStep]
    index: int = 0
    calls: list[list[BaseMessage]] = field(default_factory=list)
    tools: list[list[str]] = field(default_factory=list)
    kwargs: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a script (see the module docstring).

    Args:
        script: The steps, in order.
        label: A name used in errors (for example the role).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = "scripted"
    script: list[Any] = Field(default_factory=list, exclude=True)
    _state: _ScriptState = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        """Create the shared script state."""
        super().model_post_init(context)
        self._state = _ScriptState(steps=list(self.script))

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """The message lists of every call so far."""
        return self._state.calls

    @property
    def call_count(self) -> int:
        """How many calls the model has answered (or failed)."""
        return self._state.index

    @property
    def remaining(self) -> int:
        """How many steps of the script are left."""
        return len(self._state.steps) - self._state.index

    @property
    def bound_tools(self) -> list[list[str]]:
        """The tool names bound for each call."""
        return self._state.tools

    def extend(self, steps: Sequence[ScriptStep]) -> None:
        """Append steps to the script."""
        self._state.steps.extend(steps)

    def _next_step(self) -> ScriptStep:
        with self._state.lock:
            if self._state.index >= len(self._state.steps):
                raise ScriptExhausted(
                    f"the script of {self.label} ran out after {self._state.index} call(s)"
                )
            step = self._state.steps[self._state.index]
            self._state.index += 1
            return step

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._state.calls.append(list(messages))
        self._state.kwargs.append(dict(kwargs))
        tools = kwargs.get("tools") or []
        self._state.tools.append(
            [str(tool.get("function", {}).get("name", tool.get("name", ""))) for tool in tools]
        )
        step = self._next_step()
        if isinstance(step, BaseException):
            raise step
        message = step(list(messages)) if callable(step) else step
        if not isinstance(message, AIMessage):
            raise TypeError(f"a script step of {self.label} returned {type(message).__name__}")
        if message.id is None:
            message = message.model_copy(update={"id": f"scripted-{uuid.uuid4().hex[:12]}"})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools (recorded on each call); every other argument is accepted and ignored."""
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)


def tool_call(name: str, args: Mapping[str, Any], *, call_id: str | None = None) -> AIMessage:
    """Return an ``AIMessage`` that calls one tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": dict(args), "id": call_id or f"call_{uuid.uuid4().hex[:8]}"}
        ],
    )


def tool_calls(*calls: tuple[str, Mapping[str, Any]]) -> AIMessage:
    """Return an ``AIMessage`` that calls several tools in parallel."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": dict(args), "id": f"call_{uuid.uuid4().hex[:8]}"}
            for name, args in calls
        ],
    )


def final(text: str = "done") -> AIMessage:
    """Return an ``AIMessage`` that ends the agent's turn."""
    return AIMessage(content=text)


def structured_call(schema: type[BaseModel], data: Mapping[str, Any] | BaseModel) -> AIMessage:
    """Return the structured answer of a ``function_calling`` role (a call of the schema tool)."""
    args = (
        data.model_dump(mode="json", by_alias=True) if isinstance(data, BaseModel) else dict(data)
    )
    return tool_call(schema.__name__, args)


def json_answer(data: Mapping[str, Any] | BaseModel) -> AIMessage:
    """Return the structured answer of a ``json_schema`` role (JSON text content)."""
    payload = data.model_dump(mode="json", by_alias=True) if isinstance(data, BaseModel) else data
    return AIMessage(content=json.dumps(payload))


@dataclass(frozen=True)
class ScriptedModels(ModelRegistry):
    """A model registry of scripted models, one script per role (see the module docstring)."""

    @classmethod
    def from_scripts(
        cls,
        scripts: Mapping[Role | str, Sequence[ScriptStep]],
        *,
        default: Sequence[ScriptStep] = (),
        **kwargs: Any,
    ) -> ScriptedModels:
        """Build a registry: one :class:`ScriptedChatModel` per role, plus a default model.

        ``kwargs`` are passed to :class:`~langgraph_agentic_scientist.ModelRegistry` (structured
        method, context window, pool).
        """
        by_role: dict[Role, BaseChatModel] = {
            Role(role): ScriptedChatModel(script=list(steps), label=f"role {Role(role).value}")
            for role, steps in scripts.items()
        }
        default_model = ScriptedChatModel(script=list(default), label="the default model")
        return cls(default=default_model, by_role=by_role, **kwargs)

    def scripted(self, role: Role | str) -> ScriptedChatModel:
        """Return the scripted model of ``role`` (the default model when it has none)."""
        model = self.by_role.get(Role(role), self.default)
        if not isinstance(model, ScriptedChatModel):
            raise TypeError(f"the model of {role} is not scripted")
        return model


__all__ = [
    "ScriptExhausted",
    "ScriptStep",
    "ScriptedChatModel",
    "ScriptedModels",
    "final",
    "json_answer",
    "structured_call",
    "tool_call",
    "tool_calls",
]
