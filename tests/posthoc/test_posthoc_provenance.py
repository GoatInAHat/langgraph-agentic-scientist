"""The port's post-hoc texts and rules against the vendored upstream source."""

from __future__ import annotations
import __future__

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langgraph_agentic_scientist.posthoc import rubric_contract as ported
from langgraph_agentic_scientist.posthoc.materialize import render_analysis_task_md
from langgraph_agentic_scientist.posthoc.models import AnalysisSpec
from langgraph_agentic_scientist.prompts import PromptRegistry

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party" / "ai-scientist-v3"


def _upstream_rubric_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "upstream_posthoc_rubric_contract", UPSTREAM / "src/posthoc_pipeline/rubric_contract.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- prompts ------------------------------------------------------------------------------------


def test_bootstrap_and_executor_prompts_are_upstreams_bytes() -> None:
    registry = PromptRegistry.default()
    for name in ("posthoc_bootstrap", "posthoc_agent"):
        original = (UPSTREAM / "src/prompts" / f"{name}.md").read_text(encoding="utf-8")
        assert registry.text(f"posthoc/{name}") == original


def test_structured_prompts_differ_from_upstream_only_where_the_agent_wrote_a_file() -> None:
    registry = PromptRegistry.default()
    for name in ("posthoc_generate", "posthoc_select"):
        original = (
            (UPSTREAM / "src/prompts" / f"{name}.md").read_text(encoding="utf-8").splitlines()
        )
        port = registry.text(f"posthoc/{name}").splitlines()
        assert len(port) == len(original)
        changed = [index for index, (a, b) in enumerate(zip(original, port, strict=True)) if a != b]
        assert len(changed) == 1
        assert (
            "posthoc_bundle.json" in port[changed[0]]
            or "posthoc_selection.json" in port[changed[0]]
        )


def test_the_placeholders_of_every_prompt_are_the_ones_the_graph_fills() -> None:
    from langgraph_agentic_scientist.prompts import placeholders

    registry = PromptRegistry.default()
    assert placeholders(registry.text("posthoc/posthoc_bootstrap")) == {
        "TASK_MD",
        "EVALUATOR_SOURCE",
        "PROGRAM_SOURCE",
        "RESULTS_JSON",
    }
    assert placeholders(registry.text("posthoc/posthoc_generate")) == {
        "TASK_MD",
        "EXTRACTION_MANIFEST",
        "EVIDENCE_MANIFEST",
        "BOOTSTRAP_REPORT",
        "RESULTS_JSON",
        "EVALUATOR_SOURCE",
    }
    assert placeholders(registry.text("posthoc/posthoc_select")) == {
        "TASK_MD",
        "BOOTSTRAP_REPORT",
        "CANDIDATES_JSON",
    }
    assert placeholders(registry.text("posthoc/posthoc_agent")) == {
        "ANALYSIS_SPEC",
        "RUBRIC_CONTRACT",
        "ATTEMPT_NUMBER",
        "ERROR_CONTEXT",
    }


def test_a_value_that_looks_like_a_placeholder_is_not_expanded_again() -> None:
    registry = PromptRegistry.default()
    rendered = registry.render(
        "posthoc/posthoc_bootstrap",
        {
            "TASK_MD": "mentions {{PROGRAM_SOURCE}}",
            "EVALUATOR_SOURCE": "e",
            "PROGRAM_SOURCE": "p",
            "RESULTS_JSON": "{}",
        },
        fragments=False,
    )
    assert "mentions {{PROGRAM_SOURCE}}" in rendered


# --- the rubric contract ------------------------------------------------------------------------


def test_rubric_ids_and_descriptions_are_upstreams() -> None:
    upstream = _upstream_rubric_module()
    assert ported.REQUIRED_RUBRIC_IDS == upstream.REQUIRED_RUBRIC_IDS
    assert ported.RUBRIC_DESCRIPTIONS == upstream._RUBRIC_DESCRIPTIONS
    assert ported.build_rubric_contract("a", "fam") == upstream.build_rubric_contract("a", "fam")
    contract = upstream.build_rubric_contract("a", "fam")
    assert ported.build_target_rubrics(contract) == upstream.build_target_rubrics(contract)


def _all_pass(analysis_id: str = "a") -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "overall_pass": True,
        "rubrics": [{"id": rubric, "pass": True} for rubric in ported.REQUIRED_RUBRIC_IDS],
    }


STATUSES: list[Any] = [
    None,
    {},
    _all_pass(),
    _all_pass("other"),
    {**_all_pass(), "overall_pass": "Pass"},
    {**_all_pass(), "overall_pass": "maybe"},
    {k: v for k, v in _all_pass().items() if k != "overall_pass"},
    {**_all_pass(), "overall_pass": False},
    {"analysis_id": "a", "all_passed": "PASSED", "rubrics": [{"id": "x", "status": "fail"}]},
    {
        "analysis_id": "a",
        "overall_pass": True,
        "rubrics": [
            {"id": rubric, "pass": rubric != "claim_grounded"}
            for rubric in ported.REQUIRED_RUBRIC_IDS
        ],
    },
    {
        "analysis_id": "a",
        "overall_pass": False,
        **{rubric: {"pass": True} for rubric in ported.REQUIRED_RUBRIC_IDS},
    },
    {
        "analysis_id": "a",
        "overall_pass": True,
        "rubric_status": {rubric: {"status": "pass"} for rubric in ported.REQUIRED_RUBRIC_IDS},
    },
    {"analysis_id": "a", "overall_pass": True, "rubrics": "nope"},
    {"analysis_id": "a", "overall_pass": True, "rubrics": ["nope"]},
    {"analysis_id": "a", "overall_pass": True, "rubrics": [{"pass": True}]},
    {"analysis_id": "a", "overall_pass": True, "rubrics": [{"id": "x"}]},
    {"analysis_id": "a", "overall_pass": True, "claim_grounded": {"pass": True}},
]


@pytest.mark.parametrize("status", STATUSES)
def test_status_normalisation_and_validation_agree_with_upstream(status: Any) -> None:
    upstream = _upstream_rubric_module()
    assert ported.normalize_posthoc_rubric_status(
        status
    ) == upstream.normalize_posthoc_rubric_status(status)
    assert ported.validate_posthoc_rubric_status(
        status, "a"
    ) == upstream.validate_posthoc_rubric_status(status, "a")


# --- the generated task.md ----------------------------------------------------------------------


def _upstream_task_md_writer() -> Any:
    tree = ast.parse((UPSTREAM / "src/posthoc_pipeline/materialize.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_write_analysis_task_md"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(
        module, "upstream_materialize", "exec", flags=__future__.annotations.compiler_flag
    )
    namespace: dict[str, Any] = {"Path": Path, "AnalysisSpec": object}
    exec(code, namespace)
    return namespace["_write_analysis_task_md"]


SPECS = [
    {},
    {"hypothesis": "h", "expected_claim": "c"},
    {
        "hypothesis": "h",
        "expected_claim": "c",
        "required_artifacts": ["a.pt", "b/c.json"],
        "expected_outputs": ["o.csv"],
        "implementation_plan": ["one", "two", "three"],
    },
    {"requires_retrain": True, "implementation_plan": ["only"]},
]


@pytest.mark.parametrize("fields", SPECS)
@pytest.mark.parametrize("has_original", [True, False])
def test_the_generated_task_md_is_what_upstream_writes(
    tmp_path: Path, fields: dict[str, Any], has_original: bool
) -> None:
    spec = AnalysisSpec(
        analysis_id="weights",
        title="Weight analysis",
        family="introspection",
        analysis_dir="posthoc/analyses/weights",
        **fields,
    )
    bundle = tmp_path / "task_bundle"
    bundle.mkdir()
    if has_original:
        (bundle / "task.md").write_text("original")
    _upstream_task_md_writer()(
        bundle,
        SimpleNamespace(
            title=spec.title,
            analysis_id=spec.analysis_id,
            family=spec.family,
            requires_retrain=spec.requires_retrain,
            hypothesis=spec.hypothesis,
            expected_claim=spec.expected_claim,
            required_artifacts=spec.required_artifacts,
            expected_outputs=spec.expected_outputs,
            implementation_plan=spec.implementation_plan,
        ),
        max_attempts_per_analysis=3,
    )
    expected = (bundle / "task.md").read_text(encoding="utf-8")
    ported_text = render_analysis_task_md(
        PromptRegistry.default(), spec, max_attempts_per_analysis=3, has_original=has_original
    )
    assert ported_text == expected
    assert json.dumps(expected)  # a plain string
