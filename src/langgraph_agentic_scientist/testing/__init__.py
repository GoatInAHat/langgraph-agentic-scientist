"""Offline test support: a scripted chat model, scripted model registries and fakes.

Nothing here touches the network or a real model. Phase packages and callers use these to test
graphs end to end.
"""

from .fakes import (
    FakeEvaluator,
    FakeFigureRenderer,
    FakeFinalEvaluator,
    FakePaperToolchain,
    FakeRetriever,
    FakeTaskGate,
    MemoryBackend,
    MemoryWorkspace,
    completed_result,
    text_pdf,
)
from .fixtures import minimal_task_ref, write_minimal_task, write_program
from .scripted import (
    ScriptedChatModel,
    ScriptedModels,
    ScriptExhausted,
    final,
    json_answer,
    structured_call,
    tool_call,
    tool_calls,
)

__all__ = [
    "FakeEvaluator",
    "FakeFigureRenderer",
    "FakeFinalEvaluator",
    "FakePaperToolchain",
    "FakeRetriever",
    "FakeTaskGate",
    "MemoryBackend",
    "MemoryWorkspace",
    "ScriptExhausted",
    "ScriptedChatModel",
    "ScriptedModels",
    "completed_result",
    "final",
    "json_answer",
    "minimal_task_ref",
    "structured_call",
    "text_pdf",
    "tool_call",
    "tool_calls",
    "write_minimal_task",
    "write_program",
]
