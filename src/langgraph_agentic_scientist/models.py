"""Roles and the chat models that play them.

Upstream runs every role through a coding-agent CLI with one configured model (``agent.model``),
optionally a model per stage (``agent.models``) and a pool sampled once per idea cycle
(``agent.pool``). :class:`ModelRegistry` expresses the same with LangChain chat models: a default,
an override per :class:`Role`, and a pool. Any LangChain chat model works.

It also holds two settings that depend on the model rather than on the method:

* the structured-output method per role: ``function_calling`` (default; it works with any
  tool-calling model, through LangChain's ``ToolStrategy`` in agents and
  ``with_structured_output(method="function_calling")`` in plain calls) or ``json_schema``
  (``ProviderStrategy(strict=True)`` and ``method="json_schema"``, for servers that support
  strict JSON schema);
* the context window per role, for models whose profile lacks ``max_input_tokens``: Deep Agents
  summarises a conversation at 85 % of the window, or at a fixed 170,000 tokens when the profile
  has none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from langchain_core.language_models import BaseChatModel


class Role(StrEnum):
    """The model-driven roles of the method that the package builds (upstream R1-R38)."""

    TASK_DESIGNER = "task_designer"
    DESIGN_STRUCTURE_REPAIRER = "design_structure_repairer"
    DESIGN_JUDGE = "design_judge"
    DESIGN_PATCH_REPAIRER = "design_patch_repairer"
    DESIGN_DELTA_AUDITOR = "design_delta_auditor"
    TASK_DRAFTER = "task_drafter"
    PREFLIGHT_REPAIRER = "preflight_repairer"
    SMOKE_REPAIRER = "smoke_repairer"
    ALIGNMENT_JUDGE = "alignment_judge"
    ALIGNMENT_REPAIRER = "alignment_repairer"
    QUALIFICATION_REPAIRER = "qualification_repairer"
    IDEATOR = "ideator"
    SEARCH_AGENT = "search_agent"
    CANDIDATE_AUTHOR = "candidate_author"
    RETROFIT_AGENT = "retrofit_agent"
    VARIANT_GENERATOR = "variant_generator"
    VARIANT_REVIEWER = "variant_reviewer"
    VARIANT_EDITOR = "variant_editor"
    VARIANT_VERIFIER = "variant_verifier"
    README_SUMMARISER = "readme_summariser"
    BOOTSTRAP_AGENT = "bootstrap_agent"
    ANALYSIS_GENERATOR = "analysis_generator"
    ANALYSIS_SELECTOR = "analysis_selector"
    ANALYSIS_EXECUTOR = "analysis_executor"
    REFERENCE_PLANNER = "reference_planner"
    PAPER_STRATEGIST = "paper_strategist"
    POSTHOC_RANKER = "posthoc_ranker"
    PAPER_WRITER = "paper_writer"
    DRAFT_GAP_ANALYST = "draft_gap_analyst"
    POSTDRAFT_REFERENCE_PLANNER = "postdraft_reference_planner"
    EVIDENCE_AUDITOR = "evidence_auditor"
    PAPER_REVIEWER = "paper_reviewer"
    FIGURE_QUALITY_REPAIRER = "figure_quality_repairer"
    AI_REVIEWER = "ai_reviewer"
    AREA_CHAIR = "area_chair"
    REVISION_WRITER = "revision_writer"


StructuredMethod = Literal["function_calling", "json_schema"]


@dataclass(frozen=True)
class ModelRegistry:
    """The chat models of a run, by role.

    Attributes:
        default: The model of every role without an override.
        by_role: Overrides by role (upstream ``agent.models``).
        pool: Models sampled once per idea cycle for the ideator and the search role (upstream
            ``agent.pool``); the worker persists its choice.
        structured_output_method: Default structured-output method of structured roles.
        structured_output_by_role: Overrides of the method by role.
        context_window: Maximum input tokens of models whose profile lacks it.
        context_window_by_role: Overrides of the window by role.
    """

    default: BaseChatModel
    by_role: Mapping[Role, BaseChatModel] = field(default_factory=dict)
    pool: Sequence[BaseChatModel] = ()
    structured_output_method: StructuredMethod = "function_calling"
    structured_output_by_role: Mapping[Role, StructuredMethod] = field(default_factory=dict)
    context_window: int | None = None
    context_window_by_role: Mapping[Role, int] = field(default_factory=dict)

    def for_role(self, role: Role, *, pool_index: int | None = None) -> BaseChatModel:
        """Return the model of ``role``; with a pool index, the pool's model of the cycle."""
        if pool_index is not None and self.pool:
            return self.pool[pool_index % len(self.pool)]
        return self.by_role.get(role, self.default)

    def method_for(self, role: Role) -> StructuredMethod:
        """Return the structured-output method of ``role``."""
        return self.structured_output_by_role.get(role, self.structured_output_method)

    def window_for(self, role: Role) -> int | None:
        """Return the context window configured for ``role``, if any."""
        return self.context_window_by_role.get(role, self.context_window)


__all__ = ["ModelRegistry", "Role", "StructuredMethod"]
