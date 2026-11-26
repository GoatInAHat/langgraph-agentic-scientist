"""What a run is given: models, workspace, evaluators, prompts, configuration, hooks and policy.

:class:`ScientistContext` is LangGraph's context schema of every graph of the package. It is not
checkpointed; a resumed run is given a context again (``ainvoke(None, config, context=ctx)``).
Because a resume could pass a different context, the first node of a run records the method
configuration and the prompt texts in state and later nodes read limits from there
(:mod:`~langgraph_agentic_scientist.core.runconfig`).

Defaults that live in a phase package (``LocalTaskGate``, ``HarnessRubricVerifier``,
``SemanticScholarRetriever``, ``WorkspacePaperToolchain``) are resolved by that package when the
field is ``None``, so that the core never imports a phase.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from .agents.factory import AgentFactory, DeepAgentFactory
from .config import ScientistConfig
from .core.interfaces import (
    ConceptFigureRenderer,
    PaperToolchain,
    ReferenceRetriever,
    RubricVerifier,
    TaskGate,
)
from .core.state import Answer, Question
from .evaluation.protocol import Evaluator, FinalEvaluator
from .models import ModelRegistry
from .prompts import PromptRegistry
from .workspace.protocol import Workspace

StepHook = Callable[
    [Mapping[str, Any], "Runtime[ScientistContext]"], Awaitable[Mapping[str, Any] | None]
]
"""A hook: receives the state and the runtime; may return an update (see :class:`Hooks`)."""

SlotFunction = Callable[[Any, "Runtime[ScientistContext]"], Awaitable[Any]]
"""A slot given as an async callable ``(input, runtime) -> output``."""


@dataclass(frozen=True)
class Hooks:
    """Caller hooks before and after the steps of the method.

    Keys are step names ``"<phase>.<step>"`` (for example ``"paper.finalize"``), listed in
    ``docs/STEPS.md``. Each hook runs in its own graph node (``<phase>.<step>.before`` /
    ``.after``), so a completed hook is checkpointed and never runs again after a crash, an
    interrupt or a resume of a later node. A hook that is itself interrupted or crashes runs
    again, so a hook with an effect must be idempotent or key its effect by the step name and the
    thread (``runtime.execution_info``).

    A hook may return ``{"hook_outputs": {...}}``, which is merged under its step name, and
    updates of the keys the step declares hook-writable; any other key is refused. A hook that
    raises fails the step.
    """

    before: Mapping[str, Sequence[StepHook]] = field(default_factory=dict)
    after: Mapping[str, Sequence[StepHook]] = field(default_factory=dict)


@runtime_checkable
class HumanPolicy(Protocol):
    """Answers questions the graph would put to a person; ``None`` asks the person."""

    def decide(self, question: Question) -> Answer | None:
        """Return an answer, or ``None`` to ask the person through ``interrupt``."""
        ...


class InterruptPolicy:
    """Always asks the person: the run stops with an interrupt carrying the question."""

    def decide(self, question: Question) -> Answer | None:
        """Return ``None`` so that the question goes to the person."""
        return None


class AutoApprovePolicy:
    """Answers every question with its default, as upstream does without a terminal.

    For upstream's only question (the prune warning) the default is ``proceed``, which is what
    upstream does when standard input is not a TTY.
    """

    def decide(self, question: Question) -> Answer | None:
        """Return the question's default answer."""
        return Answer(choice=question.default, note="answered by AutoApprovePolicy")


@dataclass(frozen=True)
class GraphOverrides:
    """Replacement graphs, given when a graph is compiled.

    Attributes:
        phases: Replacement compiled graphs of whole phases, keyed by phase name.
        slots: Replacement subgraphs of replaceable steps (``investigation.ideation``,
            ``investigation.attempt``, ``investigation.worker``, ``final_evaluation.draw_inputs``,
            ``paper.data``, ``paper.references``, ``paper.postdraft_enrichment``,
            ``paper.validation``, ``paper.audit``, ``paper.review_gate``, ``paper.figures``,
            ``review``, ``revision``). A slot is a compiled graph that accepts the slot's input
            schema and returns its output schema, or an async callable ``(input, runtime) ->
            output``.
    """

    phases: Mapping[str, CompiledStateGraph[Any, Any, Any, Any]] = field(default_factory=dict)
    slots: Mapping[str, CompiledStateGraph[Any, Any, Any, Any] | SlotFunction] = field(
        default_factory=dict
    )


@dataclass(frozen=True, kw_only=True)
class ScientistContext:
    """The run context of every graph of the package.

    Attributes:
        models: The chat models by role.
        workspace: Where the run keeps its files.
        evaluator: Evaluates candidates (required by phases that evaluate).
        final_evaluator: Runs the sealed final evaluation (required by that phase only).
        task_gate: Drafting gates; ``None`` uses the drafting package's ``LocalTaskGate``.
        rubric_verifier: ``None`` uses the investigation package's ``HarnessRubricVerifier``.
        retriever: ``None`` uses the paper package's ``SemanticScholarRetriever``.
        paper_toolchain: ``None`` uses the paper package's ``WorkspacePaperToolchain``.
        figure_renderer: ``None`` renders no concept figure (the prompts' plain-export
            fallback applies).
        prompts: The prompt registry, with any caller overrides.
        config: The method configuration and the runtime settings.
        hooks: Caller hooks around the steps of the method.
        policy: Answers questions in a person's place, or asks the person.
        agent_factory: Builds the agent of each role.
    """

    models: ModelRegistry
    workspace: Workspace
    evaluator: Evaluator | None = None
    final_evaluator: FinalEvaluator | None = None
    task_gate: TaskGate | None = None
    rubric_verifier: RubricVerifier | None = None
    retriever: ReferenceRetriever | None = None
    paper_toolchain: PaperToolchain | None = None
    figure_renderer: ConceptFigureRenderer | None = None
    prompts: PromptRegistry = field(default_factory=PromptRegistry.default)
    config: ScientistConfig = field(default_factory=ScientistConfig)
    hooks: Hooks = field(default_factory=Hooks)
    policy: HumanPolicy = field(default_factory=InterruptPolicy)
    agent_factory: AgentFactory = field(default_factory=DeepAgentFactory)


__all__ = [
    "AutoApprovePolicy",
    "GraphOverrides",
    "Hooks",
    "HumanPolicy",
    "InterruptPolicy",
    "ScientistContext",
    "SlotFunction",
    "StepHook",
]
