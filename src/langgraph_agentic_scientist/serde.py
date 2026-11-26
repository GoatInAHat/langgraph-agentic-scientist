"""The msgpack allowlist for checkpointing the package's state under strict deserialization.

LangGraph's ``JsonPlusSerializer`` stores pydantic models and enums by module and class name. In
its default permissive mode it rebuilds any such type with a warning; with
``LANGGRAPH_STRICT_MSGPACK=true`` (or an explicit allowlist) it refuses every type that is not
allowlisted. :func:`serde_allowlist` returns the ``(module, class)`` pairs of every type the
package stores: every subclass of
:class:`~langgraph_agentic_scientist.core.state.StateModel` (found by importing every module of
the package and walking the subclasses, so the list cannot fall out of date) and the package's
enums. Use it with a checkpointer::

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
    async with AsyncPostgresSaver.from_conn_string(url, serde=serde) as saver:
        ...

or ``saver.with_allowlist(serde_allowlist())`` on an existing saver whose serializer is already
strict. Pass extra types (for example a caller's own structured-output schemas stored in agent
state) as arguments.
"""

from __future__ import annotations

import importlib
import pkgutil
from enum import Enum

from .core.state import StateModel


def _import_all_modules() -> None:
    package = importlib.import_module(__package__ or "langgraph_agentic_scientist")
    for module in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        if ".testing" in module.name:
            continue
        importlib.import_module(module.name)


def _subclasses(base: type) -> set[type]:
    found: set[type] = set()
    pending: list[type] = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        if cls in found:
            continue
        found.add(cls)
        pending.extend(cls.__subclasses__())
    return found


def serde_allowlist(*extra: type) -> list[tuple[str, str]]:
    """Return the ``(module, class)`` pairs of every type the package stores in state.

    Args:
        *extra: Further types to allow (for example the caller's own pydantic models).
    """
    _import_all_modules()
    types: set[type] = set(_subclasses(StateModel))
    root = __package__ or "langgraph_agentic_scientist"
    types |= {cls for cls in _subclasses(Enum) if cls.__module__.startswith(root)}
    types |= set(extra)
    return sorted(
        {(cls.__module__, cls.__qualname__) for cls in types if "<locals>" not in cls.__qualname__}
    )


__all__ = ["serde_allowlist"]
