"""Agents: how a role runs (Deep Agents over a workspace backend), its tools, skills and memory."""

from .factory import (
    SKILLS_SOURCE,
    AgentFactory,
    DeepAgentFactory,
    is_transient_provider_error,
    response_format_for,
    structured_output,
    with_context_window,
)
from .memory import (
    MemoryDelta,
    MemoryPreState,
    has_record,
    memory_delta,
    memory_pre_state,
    parse_memory_records,
    read_memory_records,
)
from .staging import SKILLS_BY_STEP, skills_for_step, stage_skills
from .tools import (
    EvaluateProgramSettings,
    make_clone_program_tool,
    make_evaluate_program_tool,
    make_file_sha256_tool,
    make_nl_tool,
    make_report_blocker_tool,
    make_task_gate_tool,
)

__all__ = [
    "SKILLS_BY_STEP",
    "SKILLS_SOURCE",
    "AgentFactory",
    "DeepAgentFactory",
    "EvaluateProgramSettings",
    "MemoryDelta",
    "MemoryPreState",
    "has_record",
    "is_transient_provider_error",
    "make_clone_program_tool",
    "make_evaluate_program_tool",
    "make_file_sha256_tool",
    "make_nl_tool",
    "make_report_blocker_tool",
    "make_task_gate_tool",
    "memory_delta",
    "memory_pre_state",
    "parse_memory_records",
    "read_memory_records",
    "response_format_for",
    "skills_for_step",
    "stage_skills",
    "structured_output",
    "with_context_window",
]
