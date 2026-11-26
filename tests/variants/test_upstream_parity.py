"""The texts, contracts and helpers of the port equal the vendored upstream code's.

The upstream variants modules cannot be imported (they import the rest of upstream's package),
so the tests execute the upstream functions that build texts and contracts, parsed from the
vendored files, and compare their results with the package's.
"""

from __future__ import annotations

import json
import random
import types
from pathlib import Path
from typing import Any

import pytest

from langgraph_agentic_scientist import PromptRegistry, VariantsConfig
from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult
from langgraph_agentic_scientist.variants import constants
from langgraph_agentic_scientist.variants import execute as ex
from langgraph_agentic_scientist.variants.cross_task import (
    write_control_cross_task_md,
    write_variant_cross_task_md,
)
from langgraph_agentic_scientist.variants.materialize import write_variant_task_md
from langgraph_agentic_scientist.variants.models import make_spec, spec_to_dict
from langgraph_agentic_scientist.variants.readme import (
    assemble_readme,
    fallback_summary,
    fallback_task_summary,
)
from langgraph_agentic_scientist.variants.retrofit import build_retrofit_prompt, format_csv_schemas
from langgraph_agentic_scientist.variants.rubrics import (
    build_rubric_contract,
    build_target_rubrics,
    canonicalize_variant_rubric_status,
    validate_variant_rubric_status,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace

from .helpers import UPSTREAM, load_upstream

PROMPTS = PromptRegistry.default()


def _constants() -> types.SimpleNamespace:
    return load_upstream(
        "constants.py",
        [
            "RUN_SUMMARY_COLUMNS",
            "METRIC_VALUES_COLUMNS",
            "TIMESERIES_COLUMNS",
            "ANALYSIS_CSV_SCHEMAS",
            "REQUIRED_RUBRIC_IDS",
            "MAX_ABLATIONS",
            "MAX_EXTRA_BASELINES",
            "MAX_EVALS_PER_VARIANT",
            "VALID_VARIANT_KINDS",
        ],
    )


def test_constants_equal_upstream() -> None:
    up = _constants()
    assert constants.RUN_SUMMARY_COLUMNS == up.RUN_SUMMARY_COLUMNS
    assert constants.METRIC_VALUES_COLUMNS == up.METRIC_VALUES_COLUMNS
    assert constants.TIMESERIES_COLUMNS == up.TIMESERIES_COLUMNS
    assert constants.ANALYSIS_CSV_SCHEMAS == up.ANALYSIS_CSV_SCHEMAS
    assert constants.REQUIRED_RUBRIC_IDS == up.REQUIRED_RUBRIC_IDS
    assert constants.VALID_VARIANT_KINDS == up.VALID_VARIANT_KINDS
    defaults = VariantsConfig()
    assert defaults.max_ablations == up.MAX_ABLATIONS == 3
    assert defaults.max_extra_baselines == up.MAX_EXTRA_BASELINES == 3
    assert defaults.max_evals_per_variant == up.MAX_EVALS_PER_VARIANT == 2


def test_variants_config_defaults_are_the_upstream_config() -> None:
    """``configs/variants_config.yaml``: caps, tries and session wall clocks."""
    text = (UPSTREAM.parents[1] / "configs" / "variants_config.yaml").read_text()
    values = {}
    for line in text.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, _, rest = line.partition(":")
            rest = rest.split("#")[0].strip()
            if rest and key.strip() and key.startswith("  "):
                values[key.strip()] = rest
    cfg = VariantsConfig()
    assert int(values["max_ablations"]) == cfg.max_ablations
    assert int(values["max_extra_baselines"]) == cfg.max_extra_baselines
    assert int(values["max_evals_per_variant"]) == cfg.max_evals_per_variant
    assert int(values["variant_timeout_seconds"]) == cfg.variant_timeout_seconds
    assert int(values["generation_timeout_seconds"]) == cfg.generation_timeout_seconds
    assert int(values["retrofit_timeout_seconds"]) == cfg.retrofit_timeout_seconds
    assert int(values["selection_timeout_seconds"]) == cfg.selection_timeout_seconds
    assert int(values["readme_timeout_seconds"]) == cfg.readme_timeout_seconds
    assert int(values["cross_task_timeout_seconds"]) == cfg.cross_task_timeout_seconds
    assert int(values["num_replications"]) == cfg.num_replications
    assert (
        cfg.edit_max_tries == cfg.verify_max_tries == 3
    )  # execute.py: _MAX_INCOMPLETE_RETRIES + 1


# --- rubric contract -----------------------------------------------------------------------------


def _upstream_rubrics() -> types.SimpleNamespace:
    up_constants = _constants()
    return load_upstream(
        "rubric_contract.py",
        [
            "_RUBRIC_DESCRIPTIONS",
            "CONTRACT_VERSION",
            "_MAX_RUBRICS",
            "_MAX_EVIDENCE_LENGTH",
            "_MAX_RECOMMENDATION_LENGTH",
            "build_rubric_contract",
            "build_target_rubrics",
            "validate_variant_rubric_status",
            "canonicalize_variant_rubric_status",
            "_validated_rubrics",
        ],
        {"REQUIRED_RUBRIC_IDS": up_constants.REQUIRED_RUBRIC_IDS},
    )


@pytest.mark.parametrize(
    ("variant_id", "kind", "base"),
    [
        ("control_best", "control", "best"),
        ("ablation_no_x", "ablation", "best"),
        ("baseline_reference", "baseline", "reference"),
        ("baseline_dfa", "baseline", "reference"),
        ("baseline_simple", "baseline", "best"),
    ],
)
def test_rubric_contract_equals_upstream(variant_id: str, kind: str, base: str) -> None:
    up = _upstream_rubrics()
    mine = build_rubric_contract(PROMPTS, variant_id, kind, base)
    assert mine == up.build_rubric_contract(variant_id, kind, base)
    assert build_target_rubrics(mine) == up.build_target_rubrics(mine)
    ids = [r["id"] for r in mine["rubrics"]]
    assert len(ids) == (12 if base == "reference" and variant_id != "baseline_reference" else 9)


def _status(rng: random.Random, ids: list[str], variant_id: str) -> dict[str, Any]:
    rubrics: list[Any] = []
    for rubric_id in ids:
        roll = rng.random()
        if roll < 0.05:
            continue
        entry: dict[str, Any] = {
            "id": rubric_id,
            "pass": rng.choice([True, True, False, "yes"]),
            "evidence": rng.choice(["ok", "", "  padded  ", "x" * 5000, 7]),
        }
        rubrics.append(entry)
        if roll > 0.97:
            rubrics.append(dict(entry))
    if rng.random() < 0.1:
        rubrics.append({"id": "extra", "pass": True, "evidence": "e"})
    status: dict[str, Any] = {
        "variant_id": rng.choice([variant_id, "other"]),
        "overall_pass": rng.choice([True, False, "x"]),
        "rubrics": rubrics if rng.random() > 0.05 else "not a list",
        "next_recommendation": rng.choice(["", "do x", 3, "y" * 5000]),
    }
    return status


def test_rubric_status_validation_and_canonicalisation_equal_upstream() -> None:
    up = _upstream_rubrics()
    rng = random.Random(7)
    ids = list(constants.REQUIRED_RUBRIC_IDS)
    for _ in range(400):
        status: dict[str, Any] | None = _status(rng, ids, "v1")
        if rng.random() < 0.03:
            status = None
        assert validate_variant_rubric_status(status, "v1") == up.validate_variant_rubric_status(
            status, "v1"
        )
        kwargs = {
            "variant_id": "v1",
            "attempt_id": "v1_attempt_01",
            "idea_id": "v1",
            "required_rubric_ids": ids,
            "combined_score": 1.5,
            "correct": rng.choice([True, False]),
        }
        assert canonicalize_variant_rubric_status(
            status, **kwargs
        ) == up.canonicalize_variant_rubric_status(status, **kwargs)


# --- texts assembled in code ---------------------------------------------------------------------


def _spec(kind: str, **fields: Any) -> Any:
    base = {
        "variant_id": "v_one",
        "title": "A Title",
        "kind": kind,
        "base_program": "best",
        "variant_dir": "variants/v_one",
        "hypothesis": "",
        "relevance_reason": "",
        "feasibility_reason": "",
        "target_components": [],
        "implementation_plan": [],
        "expected_effect": "",
        "description_body": "",
    }
    base.update(fields)
    upstream = types.SimpleNamespace(**base)
    upstream.to_dict = lambda: dict(base)  # type: ignore[attr-defined]
    mine = make_spec(**{k: v for k, v in base.items()})
    return upstream, mine


_CASES = [
    ("control", {}),
    ("control", {"hypothesis": "h", "expected_effect": "e"}),
    (
        "ablation",
        {
            "hypothesis": "drop it",
            "target_components": ["a", "b"],
            "implementation_plan": ["x", "y"],
        },
    ),
    ("ablation", {"description_body": "Long text.\n\nMore.", "expected_effect": "lower"}),
    ("baseline", {"variant_id": "baseline_reference", "base_program": "reference"}),
    ("baseline", {"variant_id": "baseline_dfa", "base_program": "reference", "hypothesis": "h"}),
    ("baseline", {"variant_id": "baseline_simple", "implementation_plan": ["one"]}),
]


@pytest.mark.parametrize(("kind", "fields"), _CASES)
@pytest.mark.parametrize("original_exists", [True, False])
async def test_variant_task_md_equals_upstream(
    tmp_path: Path,
    workspace: LocalWorkspace,
    kind: str,
    fields: dict[str, Any],
    original_exists: bool,
) -> None:
    up = load_upstream("materialize.py", ["_write_variant_task_md"])
    upstream_spec, spec = _spec(kind, **fields)
    expected_dir = tmp_path / "upstream"
    expected_dir.mkdir()
    if original_exists:
        (expected_dir / "task.md").write_text("# Original\n")
        await workspace.write_bytes("bundle/task.md", b"# Original\n")
    up._write_variant_task_md(expected_dir, upstream_spec, max_evals=3)
    await write_variant_task_md(workspace, PROMPTS, "bundle", spec, max_evals=3)
    assert (workspace.root / "bundle" / "task.md").read_text() == (
        expected_dir / "task.md"
    ).read_text()
    if original_exists:
        assert (workspace.root / "bundle" / "original_task.md").read_text() == "# Original\n"
    else:
        assert not (workspace.root / "bundle" / "original_task.md").exists()


async def test_control_cross_task_md_equals_upstream(
    tmp_path: Path, workspace: LocalWorkspace
) -> None:
    up = load_upstream(
        "cross_task.py", ["_write_control_cross_task_md"], {"_CONTROL_BEST_ID": "control_best"}
    )
    up._write_control_cross_task_md(tmp_path, "ed_conv", max_evals=4)
    await write_control_cross_task_md(workspace, PROMPTS, "bundle", "ed_conv", max_evals=4)
    assert (workspace.root / "bundle" / "task.md").read_text() == (tmp_path / "task.md").read_text()


@pytest.mark.parametrize("has_control", [True, False])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"hypothesis": "h"},
        {"hypothesis": "h", "expected_effect": "e", "description_body": "body"},
        {"expected_effect": "e", "description_body": "body"},
    ],
)
async def test_variant_cross_task_md_equals_upstream(
    tmp_path: Path, workspace: LocalWorkspace, has_control: bool, fields: dict[str, Any]
) -> None:
    up = load_upstream("cross_task.py", ["_write_variant_cross_task_md"])
    upstream_spec, spec = _spec("ablation", **fields)
    up._write_variant_cross_task_md(
        tmp_path, upstream_spec, "ed_conv", has_adapted_control=has_control, max_evals=2
    )
    await write_variant_cross_task_md(
        workspace, PROMPTS, "bundle", spec, "ed_conv", has_adapted_control=has_control, max_evals=2
    )
    assert (workspace.root / "bundle" / "task.md").read_text() == (tmp_path / "task.md").read_text()


def test_readme_assembly_and_fallbacks_equal_upstream(tmp_path: Path) -> None:
    up = load_upstream(
        "readme.py",
        [
            "_KIND_SORT_KEY",
            "_assemble_readme",
            "_fallback_summary",
            "_fallback_task_summary",
            "_read_task_md",
        ],
    )
    pairs = [
        _spec("ablation", variant_id="ablation_no_x", title="No X", hypothesis="drop x"),
        _spec("control", variant_id="control_best", title="Control (best program)"),
        _spec("baseline", variant_id="baseline_b", title="B", hypothesis=""),
        _spec("ablation", variant_id="ablation_a", title="A"),
    ]
    up_specs = [u for u, _ in pairs]
    specs = [m for _, m in pairs]
    summaries = {"control_best": "The control.", "ablation_a": ""}
    status_map = {
        "control_best": {"status": "completed", "rubric_passed": True},
        "ablation_a": {"status": "failed", "rubric_passed": False},
    }
    flat = {
        "other": [
            {"variant_id": "z", "title": "Z", "kind": "baseline", "hypothesis": "hz"},
            {"variant_id": "y"},
        ]
    }
    for task_info in (
        None,
        [{"name": "t", "data_path": "variants/", "goal": "Maximise", "role": "primary"}],
        [
            {"name": "t", "data_path": "variants/", "goal": "", "role": "primary"},
            {
                "name": "x",
                "data_path": "extra_tasks/x/variants/",
                "goal": "g",
                "role": "cross-task",
            },
            {"name": "other", "data_path": "extra_tasks/other/", "goal": "", "role": "flat"},
        ],
    ):
        for flat_tasks in ({}, flat):
            expected = up._assemble_readme(
                "Summary.",
                up_specs,
                {
                    s.variant_id: summaries.get(s.variant_id, "") or up._fallback_summary(s)
                    for s in up_specs
                },
                status_map,
                task_info,
                flat_tasks,
            )
            actual = assemble_readme(
                PROMPTS,
                "Summary.",
                specs,
                {
                    s.variant_id: summaries.get(s.variant_id, "") or fallback_summary(PROMPTS, s)
                    for s in specs
                },
                status_map,
                task_info,
                flat_tasks,
            )
            assert actual == expected
    (tmp_path / "task.md").write_text("# Heading\n\nFirst paragraph line.\nSecond.\n")
    assert fallback_task_summary(PROMPTS, "# Heading\n\nFirst paragraph line.\nSecond.\n") == (
        up._fallback_task_summary(tmp_path)
    )
    assert fallback_task_summary(PROMPTS, "# only heading\n") == up._fallback_task_summary(
        _empty_task(tmp_path)
    )


def _empty_task(tmp_path: Path) -> Path:
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "task.md").write_text("# only heading\n")
    return directory


# --- prompts built in execute.py and retrofit.py -------------------------------------------------


def test_edit_and_verify_prompts_equal_upstream() -> None:
    up = load_upstream(
        "execute.py",
        ["_build_host_edit_prompt", "_build_host_verify_prompt"],
        {
            "json": json,
            "_previous_rubric_summary": lambda bundle, index, spec: "(none; first candidate)",
        },
    )
    upstream_spec, spec = _spec(
        "ablation", hypothesis="h {{LOOKS_LIKE_A_PLACEHOLDER}}", target_components=["a"]
    )
    for program_index, attempt in [(1, 1), (2, 3)]:
        expected = up._build_host_edit_prompt(
            spec=upstream_spec,
            task_bundle_dir=Path("."),
            program_index=program_index,
            edit_attempt=attempt,
        )
        actual = ex.build_edit_prompt(
            PROMPTS,
            spec=spec,
            program_index=program_index,
            edit_attempt=attempt,
            previous_status="(none; first candidate)",
        )
        assert actual == expected
    ids = list(constants.REQUIRED_RUBRIC_IDS)
    for correct in (True, False):
        for previous in (None, "verifier exited with code 3"):
            assert ex.build_verify_prompt(
                PROMPTS,
                spec=spec,
                program_index=2,
                evaluation_correct=correct,
                target_rubric_ids=ids,
                previous_error=previous,
            ) == up._build_host_verify_prompt(
                spec=upstream_spec,
                program_index=2,
                evaluation_correct=correct,
                target_rubric_ids=ids,
                previous_error=previous,
            )


def test_failed_status_and_error_context_equal_upstream() -> None:
    up = load_upstream(
        "execute.py",
        ["_failed_rubric_status", "_variant_attempt_id", "_verification_error_context"],
        {"_MAX_VERIFICATION_ERROR_LENGTH": 256},
    )
    evaluation = types.SimpleNamespace(metrics={"combined_score": 2.5}, correct=True)
    result = EvaluationResult(
        evaluation_id="e", status="completed", metrics={"combined_score": 2.5}, correct=True
    )
    ids = list(constants.REQUIRED_RUBRIC_IDS)
    for error in (None, "boom"):
        assert ex.failed_rubric_status("v1", 2, ids, result, error) == up._failed_rubric_status(
            "v1", 2, ids, evaluation, error
        )
    text = "a  b\n c" + "x" * 400
    assert ex.verification_error_context(text) == up._verification_error_context(text)


def test_csv_schema_text_equals_upstream() -> None:
    up = load_upstream(
        "retrofit.py",
        ["_format_csv_schemas"],
        {"ANALYSIS_CSV_SCHEMAS": _constants().ANALYSIS_CSV_SCHEMAS},
    )
    assert format_csv_schemas() == up._format_csv_schemas()


async def test_retrofit_prompt_equals_upstream(tmp_path: Path, workspace: LocalWorkspace) -> None:
    template = UPSTREAM.parents[1] / "src" / "prompts" / "variant_retrofit.md"
    up = load_upstream(
        "retrofit.py",
        ["_build_retrofit_prompt", "_format_csv_schemas"],
        {
            "ANALYSIS_CSV_SCHEMAS": _constants().ANALYSIS_CSV_SCHEMAS,
            "_PROMPT_TEMPLATE_PATH": template,
            "_DEFAULT_RETROFIT_PROMPT": "",
        },
    )
    files = {
        "task.md": "# Task\n\n## Goal\nGo.\n",
        "evaluate.py": "print('evaluate')\n",
        "extraction_manifest.json": '{"a": 1}\n',
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
        await workspace.write_bytes(f"bundle/{name}", text.encode())
    for reference in ("", "## Reference Evaluator\n```python\nx\n```\n"):
        expected = up._build_retrofit_prompt(tmp_path, reference_evaluator_source=reference)
        actual = await build_retrofit_prompt(
            workspace, PROMPTS, "bundle", reference_evaluator_source=reference
        )
        assert actual == expected


def test_variant_spec_layout_matches_upstream_to_dict() -> None:
    _, spec = _spec("ablation", hypothesis="h", target_components=["c"], description_body="d")
    up = load_upstream("execute.py", ["_variant_attempt_id"])
    assert up._variant_attempt_id("v", 3) == "v_attempt_03"
    assert list(spec_to_dict(spec)) == [
        "variant_id",
        "title",
        "kind",
        "base_program",
        "variant_dir",
        "hypothesis",
        "relevance_reason",
        "feasibility_reason",
        "target_components",
        "implementation_plan",
        "expected_effect",
        "description_body",
    ]
