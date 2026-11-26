"""The README example runs: the same code, a scripted model, the phases stubbed at their edge."""

from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path
from types import ModuleType

from pipeline_stubs import Ledger, stub_overrides

from langgraph_agentic_scientist import ScientistConfig
from langgraph_agentic_scientist.config import ALL_PHASES
from langgraph_agentic_scientist.testing import ScriptedChatModel

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPO_ROOT / "examples" / "quickstart.py"
START, END = "# --8<-- [start:example]", "# --8<-- [end:example]"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quickstart_example", QUICKSTART)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snippet() -> str:
    text = QUICKSTART.read_text()
    return text.split(START, 1)[1].split(END, 1)[0].strip("\n")


async def test_the_example_runs_to_completion_and_resumes_after_an_interrupt(
    tmp_path: Path,
) -> None:
    ledger = Ledger()
    ledger.pause.add("investigation")  # the run stops once with an interrupt
    out = await _load().run(
        ScriptedChatModel(script=[]),
        tmp_path,
        config=ScientistConfig(phases=ALL_PHASES),
        overrides=stub_overrides(ledger),
    )
    assert out["status"] == "completed" and not out.get("__interrupt__")
    assert ledger.calls["investigation"] == 1 and ledger.calls["paper"] == 1
    assert (tmp_path / "run" / "task" / "evaluate.py").is_file()


async def test_the_example_defaults_to_upstreams_campaign_preset(tmp_path: Path) -> None:
    ledger = Ledger()
    out = await _load().run(
        ScriptedChatModel(script=[]), tmp_path, overrides=stub_overrides(ledger)
    )
    assert out["phases"] == list(ALL_PHASES)  # the preset runs all eight phases
    assert out["status"] == "completed"


def test_the_readme_shows_the_code_of_the_example() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert textwrap.dedent(_snippet()).strip() in [block.strip() for block in blocks]
