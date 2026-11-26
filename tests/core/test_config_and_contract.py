"""Configuration defaults and presets; the task contract's strict rules and fingerprint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from langgraph_agentic_scientist import LocalWorkspace, ScientistConfig
from langgraph_agentic_scientist.core.contract import (
    indented_public_search_payload,
    load_task_contract,
    parse_task_contract,
    public_search_payload,
    task_scaffold_fingerprint,
)
from langgraph_agentic_scientist.core.errors import TaskContractError
from langgraph_agentic_scientist.core.render import extract_markdown_section, relative_display_path
from langgraph_agentic_scientist.evaluation import (
    parse_program_spec,
    render_default_program_spec_toml,
)
from langgraph_agentic_scientist.evaluation.program_spec import entrypoint_relpath

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_defaults_are_upstreams() -> None:
    config = ScientistConfig()
    assert config.phases[0] == "task_drafting" and "review" not in config.phases
    assert config.investigation.max_idea_cycles == 20
    assert config.investigation.max_program_evals_per_attempt == 4
    assert config.investigation.idea_timeout_seconds == 3600
    assert config.drafting.alignment_max_repairs == 10
    assert config.variants.max_evals_per_variant == 2
    assert config.posthoc.primary_metric_tolerance == 0.01
    assert config.paper.max_repair_rounds == 10
    assert config.review.num_reviewers == 5 and config.review.max_completion_tokens == 16384
    assert config.agents.execute_timeout_seconds == 600
    assert config.agents.execute_max_timeout_seconds == 7200
    with pytest.raises(ValidationError):
        ScientistConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ScientistConfig.model_validate({"drafting": {"qualification_max_repairs": 9}})


def test_presets_follow_the_upstream_files() -> None:
    pipeline = ScientistConfig.preset("pipeline")
    assert pipeline.investigation.max_idea_cycles == 3
    assert pipeline.variants.num_replications == 5
    assert (pipeline.paper.min_refs, pipeline.paper.max_refs) == (25, 30)
    paper = ScientistConfig.preset("paper")
    assert paper.investigation.max_idea_cycles == 20
    assert paper.investigation.max_program_evals_per_attempt == 2
    assert paper.investigation.harness_evaluated_attempts is True
    assert paper.variants.num_replications == 1
    assert paper.final_outcome_policy == "allow_not_qualified"
    assert paper.paper.profile == "standalone" and paper.paper.two_stage is True
    assert paper.phases[-2:] == ("review", "revision")
    assert "agents" not in paper.method_dump()
    assert "sampling" not in paper.method_dump()["review"]


def test_contract_parses_and_refuses_violations() -> None:
    raw = json.loads((FIXTURES / "contract_task" / "task_contract.json").read_text())
    contract = parse_task_contract(raw)
    assert contract.method.class_name == "toy_regression"
    assert contract.protocol.search_reset_sequence == (1,)
    assert contract.compute.final.max_candidate_rpc_calls == 100
    for mutate, message in (
        (lambda value: value.update(extra=1), "unexpected field"),
        (
            lambda value: value["evaluators"].update(search="eval.py"),
            "evaluators.search must be evaluate.py",
        ),
        (lambda value: value["data"].update(search_roles=["test"]), "must not include final roles"),
        (
            lambda value: value["protocol"].update(final_seeds=[1], final_reset_sequence=[1]),
            "disjoint",
        ),
        (lambda value: value["candidate"].update(editable_paths=["../x"]), "safe"),
        (lambda value: value["compute"]["search"].update(wall_clock_seconds=10), "at least 120"),
    ):
        broken = json.loads(json.dumps(raw))
        mutate(broken)
        with pytest.raises(TaskContractError, match=message):
            parse_task_contract(broken)


async def test_contract_loads_with_digests_and_fingerprints(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "task")
    workspace = LocalWorkspace(tmp_path / "ws")
    contract = await load_task_contract(workspace, "task")
    fingerprint = await task_scaffold_fingerprint(workspace, "task")
    assert fingerprint.startswith("sha256:") and fingerprint == await task_scaffold_fingerprint(
        workspace, "task"
    )
    payload = public_search_payload(contract)
    assert set(payload) == {"candidate_interface", "search"}
    assert "final" not in json.dumps(payload)
    assert indented_public_search_payload(contract).startswith("    {")
    await workspace.write_bytes("task/resources/final/target.txt", b"9\n")
    with pytest.raises(TaskContractError, match="provenance artifact digest mismatch"):
        await load_task_contract(workspace, "task")
    await workspace.write_bytes("task/resources/final/target.txt", b"3.5\n")
    await workspace.write_bytes("task/resources/search/undeclared.txt", b"x")
    with pytest.raises(TaskContractError, match="missing provenance checksums"):
        await load_task_contract(workspace, "task")


def test_program_spec_rules() -> None:
    upstream_spec = (REPO_ROOT / "examples/circpack/program_0/program_spec.toml").read_text()
    spec = parse_program_spec(upstream_spec)
    assert parse_program_spec(render_default_program_spec_toml()) == spec
    assert entrypoint_relpath(spec) == "src/program.py"
    with pytest.raises(ValueError, match="code_root must be 'src'"):
        parse_program_spec(upstream_spec.replace('code_root = "src"', 'code_root = "lib"'))
    with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
        parse_program_spec(upstream_spec.replace('path = "program.py"', 'path = "../x.py"'))


def test_task_sections_and_display_paths() -> None:
    text = (REPO_ROOT / "examples/circpack/task.md").read_text()
    assert extract_markdown_section(text, "goal")
    assert relative_display_path("run/ideas/01", base_dir="run") == "./ideas/01"
    assert relative_display_path("run", base_dir="run") == "."
    assert relative_display_path("task", base_dir="run/paper") == "../../task"
