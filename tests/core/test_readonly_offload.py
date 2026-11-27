"""A read-only role offloads its summarised history outside its directory, which stays as it was."""

from __future__ import annotations

from deepagents.middleware.filesystem import supports_execution

from langgraph_agentic_scientist.workspace import (
    AGENT_ARTIFACTS_DIR,
    LocalWorkspace,
    ReadOnlyBackend,
)


async def test_deep_agents_files_go_to_the_artifacts_route(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("role/present.txt", b"present")
    backend = workspace.agent_backend("role", read_only=True)
    assert isinstance(backend, ReadOnlyBackend)
    assert not supports_execution(backend)
    history = f"{backend.artifacts_root}/conversation_history/session.md"
    assert (await backend.awrite(history, "older messages")).error is None
    assert (await backend.awrite("/present.txt", "changed")).error is not None
    assert (await backend.awrite("/new.txt", "new")).error is not None
    assert await workspace.read_bytes("role/present.txt") == b"present"
    assert not await workspace.exists("role/new.txt")
    assert not await workspace.exists("role/conversation_history")
    offloaded = [
        entry.path
        for entry in await workspace.list_tree(AGENT_ARTIFACTS_DIR)
        if entry.path.endswith("conversation_history/session.md")
    ]
    assert len(offloaded) == 1
