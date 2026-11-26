"""Stage 0 (the extracted bundle) and stage 4 (one workspace per variant)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_agentic_scientist import PromptRegistry
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.testing import write_program
from langgraph_agentic_scientist.variants.extract import (
    BundleValidationError,
    ExtractionError,
    extract_bundle,
)
from langgraph_agentic_scientist.variants.materialize import (
    materialize_variant_workspaces,
    reference_differs_from_best,
)
from langgraph_agentic_scientist.variants.models import VariantCandidate, VariantSelection
from langgraph_agentic_scientist.workspace import LocalWorkspace

PROMPTS = PromptRegistry.default()
META = {"entrypoint_path": "src/program.py", "combined_score": 3.0, "correct": True}


def _read(workspace: LocalWorkspace, path: str) -> str:
    return (workspace.root / path).read_bytes().decode()


async def test_extraction_builds_the_bundle(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    # what must not travel: environments, caches, runtime markers, program directories
    await workspace.write_bytes("task/.venv/lib/x.py", b"x")
    await workspace.write_bytes("task/__pycache__/a.pyc", b"x")
    await workspace.write_bytes("task/.agentic_runtime.json", b"{}")
    await workspace.write_bytes("task/capture.py", b"capture = 1\n")
    await workspace.write_bytes("runs/best/program_1/metrics.json", b"{}")  # a result file
    bundle = await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    root = workspace.root / "out/_extracted_bundle"
    assert {p.name for p in root.iterdir()} == {
        "task.md",
        "evaluate.py",
        "capture.py",
        "program_0",
        "reference_program",
        "extraction_manifest.json",
    }
    assert "SCORE = 3.0" in (root / "program_0/src/program.py").read_text()  # the best program
    assert (root / "reference_program/src/program.py").read_text() == "SCORE = 1.0\n"  # the seed
    assert not (root / "program_0/metrics.json").exists()  # the trusted clone drops results
    manifest = json.loads((root / "extraction_manifest.json").read_text())
    assert manifest["extraction_version"] == 1 and manifest["combined_score"] == 3.0
    assert (
        manifest["source_task_dir"] == "task" and manifest["source_best_program_dir"] == best.path
    )
    assert all(
        manifest[key]
        for key in ("bundle_fingerprint", "manifest_payload_fingerprint", "source_task_fingerprint")
    )
    assert bundle.has_reference and bundle.entrypoint == "src/program.py"
    assert (root / "extraction_manifest.json").read_bytes().endswith(b"}\n")


async def test_an_extracted_bundle_is_reused_when_it_matches_and_refused_when_not(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    first = await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    stamp = (workspace.root / "out/_extracted_bundle/extraction_manifest.json").read_bytes()
    again = await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    assert again.manifest == first.manifest
    assert (workspace.root / "out/_extracted_bundle/extraction_manifest.json").read_bytes() == stamp
    # the best program changed: this bundle belongs to other inputs
    await workspace.write_bytes("runs/best/program_1/src/program.py", b"SCORE = 9.0\n")
    with pytest.raises(BundleValidationError, match="source_best_program_fingerprint"):
        await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    await workspace.write_bytes(
        "runs/best/program_1/src/program.py", b"SCORE = 3.0\n# ablate_me: reheating\n"
    )
    # the bundle was edited after it was written
    await workspace.write_bytes("out/_extracted_bundle/program_0/src/program.py", b"SCORE = 0\n")
    with pytest.raises(BundleValidationError, match="fingerprint does not match"):
        await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)


async def test_a_bundle_without_a_manifest_is_an_interrupted_extraction_and_is_rebuilt(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await workspace.write_bytes("out/_extracted_bundle/half/written.txt", b"x")
    bundle = await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    assert not (workspace.root / "out/_extracted_bundle/half").exists()
    assert bundle.manifest["combined_score"] == 3.0


async def test_task_changes_are_part_of_the_provenance(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    await workspace.write_bytes("task/evaluate.py", b"print('different')\n")
    with pytest.raises(BundleValidationError, match="source_task_fingerprint"):
        await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)


async def test_extraction_errors(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    with pytest.raises(ExtractionError, match="Task directory does not exist"):
        await extract_bundle(
            workspace,
            task=task.model_copy(update={"root": "nowhere"}),
            output_dir="o",
            best=best,
            meta=META,
        )
    with pytest.raises(ExtractionError, match="Best program directory does not exist"):
        await extract_bundle(
            workspace, task=task, output_dir="o", best=ProgramRef(path="nope"), meta=META
        )
    await workspace.write_bytes("broken/program_1/src/program.py", b"x")
    with pytest.raises(ExtractionError, match="Cannot resolve entrypoint"):
        await extract_bundle(
            workspace,
            task=task,
            output_dir="o",
            best=ProgramRef(path="broken/program_1"),
            meta=META,
        )
    await workspace.remove("task/program_0")
    with pytest.raises(ExtractionError, match="missing program_0"):
        await extract_bundle(workspace, task=task, output_dir="o", best=best, meta=META)


# --- materialisation -----------------------------------------------------------------------------


async def _bundle(workspace: LocalWorkspace, task: TaskRef, best: ProgramRef) -> Any:
    bundle = await extract_bundle(workspace, task=task, output_dir="out", best=best, meta=META)
    await workspace.copy_tree("out/_extracted_bundle", "out/_retrofitted_bundle")
    return bundle


async def _materialize(
    workspace: LocalWorkspace, selection: VariantSelection | None, **kw: Any
) -> Any:
    return await materialize_variant_workspaces(
        workspace,
        PROMPTS,
        bundle_dir="out/_extracted_bundle",
        retrofitted_bundle_dir="out/_retrofitted_bundle",
        variants_run_dir="out",
        selection=selection,
        agent="test",
        agent_model="m",
        **kw,
    )


async def test_control_and_reference_baseline_are_materialised(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await workspace.write_bytes("task/notes.log", b"log")
    await workspace.write_bytes("task/extra_file.txt", b"kept")
    await _bundle(workspace, task, best)
    candidate = VariantCandidate(
        variant_id="ablation_x",
        title="X",
        kind="ablation",
        hypothesis="h",
        target_components=["c"],
        implementation_plan=["p1", "p2"],
        expected_effect="e",
    )
    specs = await _materialize(
        workspace, VariantSelection(accepted=[candidate]), max_evals_per_variant=3
    )
    assert [(s.variant_id, s.kind) for s in specs] == [
        ("control_best", "control"),
        ("baseline_reference", "baseline"),
        ("ablation_x", "ablation"),
    ]
    control = workspace.root / "out/variants/control_best"
    assert sorted(p.name for p in control.iterdir()) == [
        "rubric_contract.json",
        "run_manifest.json",
        "target_rubrics.json",
        "task_bundle",
        "variant_spec.json",
    ]
    bundle = control / "task_bundle"
    assert (bundle / "extra_file.txt").read_text() == "kept"
    assert not (bundle / "notes.log").exists()  # logs are not carried
    assert "SCORE = 3.0" in (bundle / "program_0/src/program.py").read_text()
    reference = workspace.root / "out/variants/baseline_reference/task_bundle"
    assert (reference / "program_0/src/program.py").read_text() == "SCORE = 1.0\n"  # the seed
    assert "evaluate it unchanged" in (reference / "task.md").read_text()
    assert (reference / "original_task.md").read_text().startswith("# Minimal task")
    abl = workspace.root / "out/variants/ablation_x"
    task_md = (abl / "task_bundle/task.md").read_text()
    assert "**Max program attempts:** 3" in task_md and "## Target Components\n`c`" in task_md
    assert "## Implementation Plan\n1. p1\n2. p2" in task_md
    spec = json.loads((abl / "variant_spec.json").read_text())
    assert spec["variant_dir"] == "out/variants/ablation_x" and spec["kind"] == "ablation"
    assert json.loads((abl / "run_manifest.json").read_text())["status"] == "created"
    contract = json.loads((abl / "rubric_contract.json").read_text())
    assert (abl / "task_bundle/rubric_contract.json").read_text() == (
        abl / "rubric_contract.json"
    ).read_text()
    assert [r["id"] for r in json.loads((abl / "target_rubrics.json").read_text())] == [
        r["id"] for r in contract["rubrics"]
    ]
    assert specs[2].program == ProgramRef(
        path="out/variants/ablation_x/task_bundle/program_0", entrypoint="src/program.py"
    )


async def test_no_reference_baseline_when_the_reference_equals_the_best(
    workspace: LocalWorkspace,
) -> None:
    task_ref = TaskRef(root="task")
    await workspace.write_bytes("task/task.md", b"# T\n")
    await workspace.write_bytes("task/evaluate.py", b"x\n")
    seed = await write_program(workspace, "task/program_0", "SCORE = 1.0\n")
    await write_program(workspace, "best/program_1", "SCORE = 1.0\n")
    await extract_bundle(
        workspace,
        task=task_ref,
        output_dir="out",
        best=ProgramRef(path="best/program_1"),
        meta=META,
    )
    assert not await reference_differs_from_best(workspace, "out/_retrofitted_bundle")
    await workspace.copy_tree("out/_extracted_bundle", "out/_retrofitted_bundle")
    assert not await reference_differs_from_best(workspace, "out/_extracted_bundle")
    specs = await _materialize(workspace, None)
    assert [s.variant_id for s in specs] == ["control_best"]
    assert seed.path == "task/program_0"


async def test_a_folder_baseline_is_copied_as_program_0_with_the_control_beside_it(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await _bundle(workspace, task, best)
    await workspace.write_bytes("ref_impl/model.py", b"def run(): ...\n")
    await workspace.write_bytes("ref_impl/README.md", b"Expected performance: 0.9\n")
    await workspace.write_bytes("ref_impl/baseline.md", b"---\nvariant_id: baseline_x\n---\n")
    await workspace.write_bytes("ref_impl/__pycache__/a.pyc", b"x")
    await workspace.write_bytes("ref_impl/metrics.json", b"{}")  # a stale result
    await workspace.write_bytes("ref_impl/run.log", b"log")
    candidate = VariantCandidate(
        variant_id="baseline_ref",
        title="Ref",
        kind="baseline",
        base_program="reference",
        description_body="Folder baseline",
        reference_dir="ref_impl",
    )
    specs = await _materialize(workspace, VariantSelection(accepted=[candidate]))
    spec = next(s for s in specs if s.variant_id == "baseline_ref")
    bundle = workspace.root / spec.variant_dir / "task_bundle"
    assert (bundle / "program_0/model.py").exists() and (bundle / "program_0/README.md").exists()
    assert not (bundle / "program_0/baseline.md").exists()
    assert not (bundle / "program_0/__pycache__").exists()
    assert (
        not (bundle / "program_0/metrics.json").exists()
        and not (bundle / "program_0/run.log").exists()
    )
    assert (
        "Auto-generated for folder-based reference implementation"
        in (bundle / "program_0/program_spec.toml").read_text()
    )
    assert "SCORE = 3.0" in (bundle / "control_program/src/program.py").read_text()
    text = (bundle / "task.md").read_text()
    assert "### Step 1: Hyperparameter Audit" in text and "`control_program/`" in text
    ids = [
        r["id"]
        for r in json.loads(
            (workspace.root / spec.variant_dir / "rubric_contract.json").read_text()
        )["rubrics"]
    ]
    assert ids[-3:] == [
        "hyperparameter_audit_complete",
        "reference_conditions_aligned",
        "reference_performance_check",
    ]
    assert len(ids) == 12


async def test_a_folder_baseline_without_documentation_is_refused(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await _bundle(workspace, task, best)
    await workspace.write_bytes("ref_impl/model.py", b"x")
    candidate = VariantCandidate(
        variant_id="baseline_ref",
        title="Ref",
        kind="baseline",
        base_program="reference",
        reference_dir="ref_impl",
    )
    with pytest.raises(ValueError, match="requires documentation in program_0"):
        await _materialize(workspace, VariantSelection(accepted=[candidate]))


async def test_custom_only_reserved_ids_and_required_controls(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef]
) -> None:
    task, best = study
    await _bundle(workspace, task, best)
    candidate = VariantCandidate(variant_id="baseline_a", title="A", kind="baseline")
    only = await _materialize(workspace, VariantSelection(accepted=[candidate]), custom_only=True)
    assert [s.variant_id for s in only] == ["baseline_a"]
    with_control = await _materialize(
        workspace,
        VariantSelection(accepted=[candidate]),
        custom_only=True,
        required_controls=["control_best"],
    )
    assert [s.variant_id for s in with_control] == ["control_best", "baseline_a"]
    reserved = VariantCandidate(variant_id="control_best", title="C", kind="baseline")
    with pytest.raises(ValueError, match="reserved variant IDs: control_best"):
        await _materialize(workspace, VariantSelection(accepted=[reserved]))
    with pytest.raises(ValueError, match="Unsupported required controls"):
        await _materialize(workspace, None, required_controls=["baseline_reference"])
