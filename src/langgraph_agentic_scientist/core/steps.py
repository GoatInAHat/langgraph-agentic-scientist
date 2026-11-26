"""Building phase graphs: steps with hook nodes, replaceable slots, and questions to a person.

:class:`StepGraph` is a thin builder over LangGraph's ``StateGraph``. Its :meth:`StepGraph.step`
adds a step of the method together with two hook nodes, ``<phase>.<step>.before`` and
``<phase>.<step>.after``, wired ``before -> step -> after``. Edges are written between step
names; the builder routes an edge into a step to its ``before`` node and an edge out of a step
from its ``after`` node. Because hooks run in their own nodes, a completed hook is checkpointed
and never runs again after a crash, an interrupt or a resume of a later node (design rule D6).

Rules for phase builders:

* Route out of a hooked step with edges or conditional edges, not with ``Command(goto=...)``
  returned by the step itself (its ``after`` node would run as well).
* A step that receives ``Send`` arguments gets no hook nodes of its own (the ``before`` node would
  receive the argument, not the step); make it a subgraph whose inner steps carry the hooks.
* Pure routing nodes need no hooks: add them with :meth:`StepGraph.add_node`.

:func:`ask_person` is the one way a node asks a person: the context's
:class:`~langgraph_agentic_scientist.HumanPolicy` answers, or the run stops with an ``interrupt``
whose value is the question.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Send, interrupt

from ..context import GraphOverrides, ScientistContext
from .errors import ScientistError
from .state import Answer, Question


class HookError(ScientistError):
    """A hook returned an update of a key its step does not declare hook-writable."""


NodeFunction = Callable[..., Any]


def _is_graph(value: object) -> bool:
    return isinstance(value, Runnable | CompiledStateGraph)


def _hook_node(kind: str, step_key: str, writable: frozenset[str]) -> NodeFunction:
    async def run_hooks(
        state: Mapping[str, Any], runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        registry = runtime.context.hooks.before if kind == "before" else runtime.context.hooks.after
        hooks = registry.get(step_key, ())
        if not hooks:
            return {}
        update: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        for hook in hooks:
            result = await hook(state, runtime)
            if not result:
                continue
            for key, value in result.items():
                if key == "hook_outputs":
                    if not isinstance(value, Mapping):
                        raise HookError(f"hook of {step_key} returned non-mapping hook_outputs")
                    outputs.update(value)
                elif key in writable:
                    update[key] = value
                else:
                    raise HookError(
                        f"a {kind} hook of {step_key} may not update {key!r}; hook-writable keys "
                        f"of this step: {sorted(writable) or 'none'}"
                    )
        if outputs:
            existing = dict(state.get("hook_outputs") or {}).get(step_key) or {}
            update["hook_outputs"] = {step_key: {**dict(existing), **outputs}}
        return update

    run_hooks.__name__ = f"{kind}_hooks"
    return run_hooks


def _slot_node(
    impl: Callable[[Any, Runtime[ScientistContext]], Awaitable[Any]],
    input_mapper: Callable[[Mapping[str, Any]], Any] | None,
    output_mapper: Callable[[Any], Mapping[str, Any]] | None,
) -> NodeFunction:
    async def run_slot(
        state: Mapping[str, Any], runtime: Runtime[ScientistContext]
    ) -> Mapping[str, Any]:
        value = input_mapper(state) if input_mapper else state
        result = await impl(value, runtime)
        if output_mapper:
            return output_mapper(result)
        return dict(result or {})

    return run_slot


class StepGraph:
    """Builds a phase graph whose steps carry hook nodes and whose slots are replaceable.

    Args:
        phase: The phase name; step names are ``"<phase>.<step>"``.
        state_schema: The phase state (a ``TypedDict`` with reducers).
        input_schema: The phase input schema.
        output_schema: The phase output schema.
        overrides: Replacement slots (``GraphOverrides.slots``).
    """

    def __init__(
        self,
        phase: str,
        state_schema: type[Any],
        *,
        input_schema: type[Any] | None = None,
        output_schema: type[Any] | None = None,
        overrides: GraphOverrides | None = None,
    ) -> None:
        self.phase = phase
        self.overrides = overrides or GraphOverrides()
        self.graph: StateGraph[Any, Any, Any, Any] = StateGraph(
            state_schema,
            context_schema=ScientistContext,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        self._entries: dict[str, str] = {}
        self._exits: dict[str, str] = {}
        self.steps: list[str] = []
        self.slots: list[str] = []

    # --- names ----------------------------------------------------------------------------

    def step_key(self, name: str) -> str:
        """Return the hook key of a step: ``"<phase>.<step>"``."""
        return f"{self.phase}.{name}"

    def entry(self, name: str) -> str:
        """Return the node that an edge into step ``name`` must target."""
        if name in (START, END):
            return name
        return self._entries.get(name, name)

    def exit(self, name: str) -> str:
        """Return the node from which edges out of step ``name`` leave."""
        if name in (START, END):
            return name
        return self._exits.get(name, name)

    # --- nodes ----------------------------------------------------------------------------

    def add_node(self, name: str, action: Any, **kwargs: Any) -> str:
        """Add a node without hook nodes (routing nodes, ``Send`` targets)."""
        self.graph.add_node(name, action, **kwargs)
        return name

    def step(
        self,
        name: str,
        action: Any,
        *,
        hook_writable: Sequence[str] = (),
        hooks: bool = True,
        **kwargs: Any,
    ) -> str:
        """Add a step of the method and, unless ``hooks=False``, its before and after hook nodes.

        Args:
            name: The step name (unique in the phase).
            action: The node function or a compiled graph.
            hook_writable: State keys that a hook of this step may update.
            hooks: Whether to add hook nodes.
            **kwargs: Passed to ``StateGraph.add_node`` (``retry_policy``, ``metadata``, ...).
        """
        self.graph.add_node(name, action, **kwargs)
        self.steps.append(name)
        if not hooks:
            return name
        key = self.step_key(name)
        before = f"{key}.before"
        after = f"{key}.after"
        writable = frozenset(hook_writable)
        self.graph.add_node(before, _hook_node("before", key, writable))
        self.graph.add_node(after, _hook_node("after", key, writable))
        self.graph.add_edge(before, name)
        self.graph.add_edge(name, after)
        self._entries[name] = before
        self._exits[name] = after
        return name

    def slot(
        self,
        name: str,
        default: CompiledStateGraph[Any, Any, Any, Any]
        | Callable[[Any, Runtime[ScientistContext]], Awaitable[Any]],
        *,
        key: str | None = None,
        input_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
        output_mapper: Callable[[Any], Mapping[str, Any]] | None = None,
        hooks: bool = True,
        hook_writable: Sequence[str] = (),
        **kwargs: Any,
    ) -> str:
        """Mount a replaceable step: the override of slot ``key`` if given, else ``default``.

        A slot is a compiled graph sharing the state keys of its input and output schemas, or an
        async callable ``(input, runtime) -> output`` that the builder wraps in a node. The
        mappers translate state to the callable's input and its output to a state update.
        """
        slot_key = key or self.step_key(name)
        impl = self.overrides.slots.get(slot_key, default)
        if _is_graph(impl):
            action: Any = impl
        elif callable(impl) and inspect.iscoroutinefunction(impl):
            action = _slot_node(impl, input_mapper, output_mapper)
        elif callable(impl):
            target = impl

            async def call_sync(value: Any, runtime: Runtime[ScientistContext]) -> Any:
                result = target(value, runtime)
                return await result if inspect.isawaitable(result) else result

            action = _slot_node(call_sync, input_mapper, output_mapper)
        else:
            raise TypeError(f"slot {slot_key} must be a compiled graph or a callable")
        self.slots.append(slot_key)
        return self.step(name, action, hooks=hooks, hook_writable=hook_writable, **kwargs)

    # --- edges ----------------------------------------------------------------------------

    def add_edge(self, start: str | Sequence[str], end: str) -> None:
        """Add an edge between steps (or nodes); hook nodes are routed around."""
        if isinstance(start, str):
            self.graph.add_edge(self.exit(start), self.entry(end))
        else:
            self.graph.add_edge([self.exit(item) for item in start], self.entry(end))

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Any],
        path_map: Mapping[Hashable, str] | Sequence[str] | None = None,
    ) -> None:
        """Add conditional edges out of a step; destinations are steps or nodes.

        With a key-to-step ``path_map`` the router returns keys and the map's steps are routed to
        their entry nodes. Without one, the router may return step names or ``Send`` objects,
        which are routed to the target's entry node.
        """
        if isinstance(path_map, Mapping):
            mapped = {key: self.entry(value) for key, value in path_map.items()}
            self.graph.add_conditional_edges(self.exit(source), path, mapped)
            return

        def remap(value: Any) -> Any:
            if isinstance(value, Send):
                return Send(self.entry(value.node), value.arg)
            if isinstance(value, str):
                return self.entry(value)
            return value

        def remap_all(result: Any) -> Any:
            if isinstance(result, list | tuple):
                return [remap(item) for item in result]
            return remap(result)

        routed: Callable[..., Any]
        if inspect.iscoroutinefunction(path):
            async_path = path

            async def routed_async(*args: Any, **kwargs: Any) -> Any:
                return remap_all(await async_path(*args, **kwargs))

            routed = routed_async
        else:
            sync_path = path

            def routed_sync(*args: Any, **kwargs: Any) -> Any:
                return remap_all(sync_path(*args, **kwargs))

            routed = routed_sync
        _copy_signature(routed, path)
        targets = None if path_map is None else [self.entry(value) for value in path_map]
        self.graph.add_conditional_edges(self.exit(source), routed, targets)

    def compile(
        self,
        *,
        checkpointer: Checkpointer = None,
        store: Any = None,
        name: str | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Compile the graph (pass the checkpointer and store only at the top level)."""
        return self.graph.compile(checkpointer=checkpointer, store=store, name=name or self.phase)


def _copy_signature(wrapper: Callable[..., Any], original: Callable[..., Any]) -> None:
    """Give ``wrapper`` the router's signature with its annotations resolved.

    LangGraph reads a router's annotations to know what it receives. A router written under
    ``from __future__ import annotations`` carries them as strings, which would be resolved in
    this module's globals once copied; they are resolved in the router's own module instead.
    """
    try:
        signature = inspect.signature(original, eval_str=True)
    except (NameError, TypeError, ValueError, SyntaxError):
        try:
            signature = inspect.signature(original)
        except (TypeError, ValueError):
            return
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
    }


async def ask_person(question: Question, runtime: Runtime[ScientistContext]) -> Answer:
    """Ask a person, unless the context's policy answers in the person's place.

    The question goes out as ``interrupt(question.model_dump(mode="json"))``; the run is resumed
    with ``Command(resume=answer)``, where ``answer`` is a choice string or an
    :class:`~langgraph_agentic_scientist.core.state.Answer` mapping. ``ask_person`` must be the
    first and only effectful call of its node: on resume the node runs again from its start.

    Raises:
        ValueError: When the answer is not one of the question's options.
    """
    answer = runtime.context.policy.decide(question)
    if answer is None:
        value = interrupt(question.model_dump(mode="json"))
        if isinstance(value, Answer):
            answer = value
        elif isinstance(value, str):
            answer = Answer(choice=value)
        else:
            answer = Answer.model_validate(value)
    if question.options and answer.choice not in question.options:
        raise ValueError(
            f"answer {answer.choice!r} to {question.kind!r} is not one of {question.options}"
        )
    return answer


__all__ = ["HookError", "NodeFunction", "StepGraph", "ask_person"]
