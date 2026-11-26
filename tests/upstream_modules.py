from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ai-scientist-v3"
PACKAGES = ("src", "src.investigation", "src.variants_pipeline", "src.posthoc_pipeline")

Stubs = Mapping[str, Mapping[str, Any]]


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(UPSTREAM_ROOT.joinpath(*name.split(".")))]
    return module


@pytest.fixture
def upstream_modules() -> Iterator[Callable[..., types.ModuleType]]:
    """Import vendored upstream modules by dotted name (``src.variants_pipeline.aggregate``).

    Only the vendored files exist; ``stubs`` maps further dotted module names to the attributes
    that the imported modules take from them.
    """
    saved = {
        name: sys.modules.get(name) for name in list(sys.modules) if name.split(".")[0] == "src"
    }

    def load(name: str, stubs: Stubs | None = None) -> types.ModuleType:
        for package in PACKAGES:
            if package not in sys.modules:
                sys.modules[package] = _package(package)
        for stub_name, attributes in (stubs or {}).items():
            stub = types.ModuleType(stub_name)
            stub.__dict__.update(attributes)
            sys.modules[stub_name] = stub
        return importlib.import_module(name)

    yield load
    for name in [name for name in sys.modules if name.split(".")[0] == "src"]:
        del sys.modules[name]
    sys.modules.update({name: module for name, module in saved.items() if module is not None})
