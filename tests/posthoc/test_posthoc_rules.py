"""The post-hoc rules: hard constraints, normalisation, checkpoints, metrics, evidence, sealing."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest
from posthoc_support import ablations as _ablations

from langgraph_agentic_scientist.config import PosthocConfig
from langgraph_agentic_scientist.core.contract import load_task_contract
from langgraph_agentic_scientist.core.state import TaskRef
from langgraph_agentic_scientist.posthoc.common import (
    compare_metric_guardrail,
    find_checkpoint_candidates,
    infer_primary_metric,
    validate_checkpoint_loadable,
)
from langgraph_agentic_scientist.posthoc.extract import (
    BundleValidationError,
    extract_bundle,
    restore_verified_evidence,
    stage_evidence,
    verify_staged_evidence,
)
from langgraph_agentic_scientist.posthoc.models import (
    AnalysisCandidate,
    SelectionAnswer,
    normalize_expected_outputs,
    normalize_implementation_plan,
    normalize_string_list,
)
from langgraph_agentic_scientist.posthoc.select import (
    apply_selection_constraints,
    decisions_from_answer,
)
from langgraph_agentic_scientist.testing import write_minimal_task, write_program

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"


def _candidate(analysis_id: str, **fields: Any) -> AnalysisCandidate:
    base: dict[str, Any] = {
        "id": analysis_id,
        "title": f"title of {analysis_id}",
        "family": f"family-{analysis_id}",
        "required_artifacts": [],
        "implementation_plan": ["inspect the saved weights"],
        "expected_outputs": ["out.csv"],
    }
    return AnalysisCandidate(**{**base, **fields})


async def _constrain(
    workspace: Any,
    candidates: list[AnalysisCandidate],
    *,
    decisions: dict[str, tuple[bool, str]] | None = None,
    cap: int = 3,
    ready: bool = True,
) -> Any:
    return await apply_selection_constraints(
        workspace,
        candidates=candidates,
        llm_decisions=decisions or {},
        max_selected_analyses=cap,
        bundle_dir="ready",
        bootstrap_ready=ready,
    )


# --- the hard constraints -----------------------------------------------------------------------


async def test_hard_constraints_reject_what_the_selector_accepted(workspace: Any) -> None:
    await workspace.write_bytes("ready/program_0/model.pkl", b"x")
    await workspace.write_bytes("ready/reference_program/notes.txt", b"x")
    candidates = [
        _candidate("ablate", family="Ablation Study"),
        _candidate("retrain", requires_retrain=True),
        _candidate("plan", implementation_plan=["first Fine-tune the model"]),
        _candidate("rerun", implementation_plan=["then run evaluation again"]),
        _candidate("missing", required_artifacts=["program_0/absent.pt"]),
        _candidate(
            "ok", required_artifacts=["model.pkl", "task_bundle/reference_program/notes.txt"]
        ),
        _candidate("same-family", family="family-ok", title="another"),
        _candidate("rejected-by-llm"),
    ]
    selection = await _constrain(
        workspace, candidates, decisions={"rejected-by-llm": (False, "shallow")}
    )
    assert [item.analysis_id for item in selection.accepted] == ["ok"]
    reasons = {item.candidate.analysis_id: item.reason for item in selection.rejected}
    assert reasons["ablate"].startswith("post-hoc analysis pipeline excludes ablations")
    assert reasons["retrain"].startswith("post-hoc analysis pipeline is analysis-only;")
    assert "candidates must not require retraining" in reasons["retrain"]
    assert "use existing artifacts rather than retraining" in reasons["plan"]
    assert "use existing artifacts rather than retraining" in reasons["rerun"]
    assert reasons["missing"] == "missing required artifacts: ['program_0/absent.pt']"
    assert reasons["same-family"] == "family already represented"
    assert reasons["rejected-by-llm"] == "shallow"


async def test_retrain_reason_depends_on_the_bootstrap(workspace: Any) -> None:
    selection = await _constrain(
        workspace, [_candidate("retrain", requires_retrain=True)], ready=False
    )
    assert "bootstrap is not relevant for retraining candidates" in selection.rejected[0].reason


async def test_duplicates_and_the_cap(workspace: Any) -> None:
    candidates = [
        _candidate("a", family="x", title="Same"),
        _candidate("a", family="y"),
        _candidate("b", family="x", title=" same "),
        _candidate("c", family="y"),
        _candidate("d", family="z"),
        _candidate("e", family="w"),
    ]
    selection = await _constrain(workspace, candidates, cap=2)
    assert [item.analysis_id for item in selection.accepted] == ["a", "c"]
    reasons = [item.reason for item in selection.rejected]
    assert reasons == [
        "duplicate analysis_id",
        "duplicate family/title proposal",
        "max_selected_analyses cap reached",
        "max_selected_analyses cap reached",
    ]
    assert (await _constrain(workspace, candidates, cap=0)).accepted == []


async def test_a_required_artifact_behind_a_symlink_or_outside_never_exists(
    workspace: Any, tmp_path: Path
) -> None:
    from langgraph_agentic_scientist.posthoc.select import artifact_exists

    await workspace.write_bytes("ready/program_0/real.txt", b"x")
    await workspace.write_bytes("elsewhere/secret.txt", b"x")
    (tmp_path / "ws" / "ready" / "program_0" / "link").symlink_to(tmp_path / "ws" / "elsewhere")
    assert await artifact_exists(workspace, "ready", "real.txt")
    assert await artifact_exists(workspace, "ready", "task_bundle/program_0/real.txt")
    assert not await artifact_exists(workspace, "ready", "link/secret.txt")
    assert not await artifact_exists(workspace, "ready", "../elsewhere/secret.txt")
    assert not await artifact_exists(workspace, "ready", "/etc/passwd")
    assert not await artifact_exists(workspace, "ready", "  ")


def test_selector_decisions_default_to_accepted() -> None:
    answer = SelectionAnswer.model_validate(
        {"decisions": [{"id": "a", "accepted": False, "reason": "dup"}, {"id": "b"}]}
    )
    decisions = decisions_from_answer(answer, [_candidate("a"), _candidate("b"), _candidate("c")])
    assert decisions == {
        "a": (False, "dup"),
        "b": (True, ""),
        "c": (True, "accepted (not reviewed by selector)"),
    }


# --- normalisation ------------------------------------------------------------------------------


def test_list_like_fields_never_explode_into_characters() -> None:
    assert normalize_string_list("one") == ["one"]
    assert normalize_string_list(None) == [] and normalize_string_list("  ") == []
    assert normalize_string_list([" a ", "", 3]) == ["a", "3"]


def test_prose_outputs_become_safe_paths() -> None:
    assert normalize_expected_outputs(["plots/loss.png", "../x.csv", "A histogram of weights"]) == [
        "plots/loss.png",
        "outputs/02_x_csv.md",
        "outputs/03_a_histogram_of_weights.png",
    ]
    assert normalize_expected_outputs("a results table") == ["outputs/01_a_results_table.csv"]
    assert normalize_expected_outputs(["/abs/path.csv"])[0].startswith("outputs/01_abs_path_csv")


def test_implementation_plans_as_strings_and_character_arrays() -> None:
    assert normalize_implementation_plan("1. load it\n2. plot it") == ["load it", "plot it"]
    assert normalize_implementation_plan("1) load it 2) plot it") == ["load it", "plot it"]
    assert normalize_implementation_plan("- a\n- b") == ["a", "b"]
    characters = list("load the checkpoint and plot")
    assert normalize_implementation_plan(characters) == ["load the checkpoint and plot"]
    assert normalize_implementation_plan(["a", "", " b "]) == ["a", "b"]


# --- checkpoints --------------------------------------------------------------------------------


async def test_checkpoint_candidates_follow_results_json_then_names(workspace: Any) -> None:
    await workspace.write_bytes("p/_results.json", b'{"checkpoint_path": "out/final.bin", "x": 1}')
    await workspace.write_bytes("p/out/final.bin", b"123")
    await workspace.write_bytes("p/models/best_model.pt", b"1234")
    await workspace.write_bytes("p/data/train_images.npy", b"x")  # excluded by name
    await workspace.write_bytes("p/ckpt/other.pt", b"12")  # a checkpoint directory
    await workspace.write_bytes("p/notes.txt", b"x")
    found = await find_checkpoint_candidates(workspace, "p")
    assert [path for path, _ in found] == ["out/final.bin", "ckpt/other.pt", "models/best_model.pt"]
    assert await find_checkpoint_candidates(workspace, "absent") == []


async def test_a_checkpoint_named_outside_the_program_is_ignored(workspace: Any) -> None:
    await workspace.write_bytes("p/_results.json", b'{"model_path": "../q/model.pt"}')
    assert await find_checkpoint_candidates(workspace, "p") == []


def _safetensors(header: bytes) -> bytes:
    return struct.pack("<Q", len(header)) + header


async def test_checkpoints_are_validated_without_deserialising_objects(
    workspace: Any, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    (root / "c").mkdir()
    (root / "c" / "ok.pkl").write_bytes(pickle.dumps({"a": 1}))
    (root / "c" / "bad.pkl").write_bytes(b"not a pickle at all")
    (root / "c" / "empty.pt").write_bytes(b"")
    with zipfile.ZipFile(root / "c" / "archive.pt", "w") as archive:
        archive.writestr("data.pkl", b"x")
    with zipfile.ZipFile(root / "c" / "none.pt", "w"):
        pass
    (root / "c" / "ok.safetensors").write_bytes(_safetensors(b'{"w": {}}'))
    (root / "c" / "bad.safetensors").write_bytes(b"short")
    (root / "c" / "weights.npz").write_bytes(b"not a zip")
    (root / "c" / "ok.npy").write_bytes(b"\x93NUMPY" + b"\x01\x00" + b"v")
    (root / "c" / "doc.json").write_text("{}")
    expectations = {
        "ok.pkl": True,
        "bad.pkl": False,
        "empty.pt": False,
        "archive.pt": True,
        "none.pt": False,
        "ok.safetensors": True,
        "bad.safetensors": False,
        "weights.npz": False,
        "doc.json": True,
        "missing.pt": False,
    }
    for name, loadable in expectations.items():
        got, error = await validate_checkpoint_loadable(workspace, f"c/{name}")
        assert got is loadable, (name, error)
        assert (error is None) is loadable
    _, empty = await validate_checkpoint_loadable(workspace, "c/empty.pt")
    assert empty == "checkpoint file is empty"
    _, missing = await validate_checkpoint_loadable(workspace, "c/missing.pt")
    assert missing == "checkpoint file missing"


# --- the primary metric -------------------------------------------------------------------------


def test_primary_metric_inference_and_the_guardrail() -> None:
    metrics = {"combined_score": 3.0, "val_accuracy": 0.8, "training_time": 10}
    assert infer_primary_metric(metrics=metrics)["key"] == "val_accuracy"
    assert infer_primary_metric(metrics=metrics, requested_key="combined_score")["value"] == 3.0
    only_combined = infer_primary_metric(metrics={"combined_score": 3.0})
    assert only_combined["key"] == "combined_score"
    from_results = infer_primary_metric(metrics={}, results={"my_loss": 0.5, "seed": 1})
    assert from_results["key"] == "my_loss" and from_results["higher_is_better"] is False
    with pytest.raises(RuntimeError, match="Unable to resolve a numeric primary metric"):
        infer_primary_metric(metrics={"note": "x"})

    baseline = infer_primary_metric(metrics={"val_accuracy": 0.8})
    ok = compare_metric_guardrail(
        baseline=baseline,
        candidate=infer_primary_metric(metrics={"val_accuracy": 0.795}),
        tolerance=0.01,
    )
    assert ok["passed"] is True and ok["reason"] is None
    bad = compare_metric_guardrail(
        baseline=baseline,
        candidate=infer_primary_metric(metrics={"val_accuracy": 0.7}),
        tolerance=0.01,
    )
    assert bad["passed"] is False and bad["reason"] == "primary metric regressed beyond tolerance"
    lower = infer_primary_metric(metrics={"loss": 0.5})
    worse = infer_primary_metric(metrics={"loss": 0.6})
    assert (
        compare_metric_guardrail(baseline=lower, candidate=worse, tolerance=0.01)["passed"] is False
    )
    mismatch = compare_metric_guardrail(baseline=baseline, candidate=lower, tolerance=0.01)
    assert mismatch["reason"] == "primary metric mismatch: baseline=val_accuracy candidate=loss"


# --- evidence staging ---------------------------------------------------------------------------


async def test_evidence_is_staged_with_a_hash_manifest_and_verified(workspace: Any) -> None:
    await _ablations(workspace)
    manifest = await stage_evidence(workspace, "ablations", "bundle", PosthocConfig())
    names = sorted(manifest["files"])
    assert names == [
        "evidence/README.md",
        "evidence/aggregated/task/run_summary.csv",
        "evidence/generation/variant_bundle.json",
        "evidence/pipeline_summary.json",
        "evidence/variants/control_best/task_bundle/program_0/evaluation_evidence.json",
    ]
    assert manifest["source_role"] == "completed_ablations_bundle"
    csv = b"a,b\n1,2\n"
    assert manifest["files"]["evidence/aggregated/task/run_summary.csv"] == {
        "size": len(csv),
        "sha256": hashlib.sha256(csv).hexdigest(),
    }
    assert manifest["total_bytes"] == sum(item["size"] for item in manifest["files"].values())
    assert await verify_staged_evidence(workspace, "bundle") == manifest

    await workspace.write_bytes("bundle/evidence/README.md", b"# changed\n")
    with pytest.raises(BundleValidationError, match=r"digest mismatch|size mismatch"):
        await verify_staged_evidence(workspace, "bundle")


async def test_changed_evidence_is_detected_and_pristine_evidence_is_restored(
    workspace: Any,
) -> None:
    await _ablations(workspace)
    await stage_evidence(workspace, "ablations", "bundle", PosthocConfig())
    await restore_verified_evidence(workspace, "bundle", "ready")
    assert await verify_staged_evidence(workspace, "ready")
    assert (await workspace.read_bytes("ready/evidence/README.md")) == b"# ablations\n"
    await workspace.write_bytes("ready/evidence/extra.txt", b"smuggled")
    with pytest.raises(BundleValidationError, match="does not match manifest"):
        await verify_staged_evidence(workspace, "ready")
    await workspace.remove("ready/evidence/extra.txt")
    await workspace.remove("ready/evidence/README.md")
    with pytest.raises(BundleValidationError, match="does not match manifest"):
        await verify_staged_evidence(workspace, "ready")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"schema_version": 2}, "unsupported schema"),
        ({"total_completed": 0}, "incomplete"),
        ({"total_variants": 3}, "incomplete"),
        ({"total_completed": 2}, "inconsistent"),
    ],
)
async def test_the_evidence_anchor_must_be_a_completed_summary(
    workspace: Any, edit: dict[str, Any], message: str
) -> None:
    await _ablations(workspace)
    summary = json.loads((await workspace.read_bytes("ablations/pipeline_summary.json")).decode())
    summary.update(edit)
    await workspace.write_bytes("ablations/pipeline_summary.json", json.dumps(summary).encode())
    with pytest.raises(BundleValidationError, match=message):
        await stage_evidence(workspace, "ablations", "bundle", PosthocConfig())


async def test_evidence_limits_and_invalid_evidence_are_enforced(workspace: Any) -> None:
    await _ablations(workspace)
    with pytest.raises(BundleValidationError, match="file count exceeds"):
        await stage_evidence(workspace, "ablations", "b1", PosthocConfig(evidence_max_files=2))
    with pytest.raises(BundleValidationError, match="file exceeds safety limit"):
        await stage_evidence(workspace, "ablations", "b2", PosthocConfig(evidence_max_file_bytes=5))
    with pytest.raises(BundleValidationError, match="bundle exceeds safety limit"):
        await stage_evidence(
            workspace, "ablations", "b3", PosthocConfig(evidence_max_total_bytes=40)
        )
    await workspace.write_bytes(
        "ablations/variants/control_best/task_bundle/program_0/evaluation_evidence.json",
        b'{"schema_version": 1, "episodes": []}',
    )
    with pytest.raises(BundleValidationError, match="Invalid evaluation evidence"):
        await stage_evidence(workspace, "ablations", "b4", PosthocConfig())
    with pytest.raises(BundleValidationError, match="not a real directory"):
        await stage_evidence(workspace, "nowhere", "b5", PosthocConfig())


# --- the extracted bundle -----------------------------------------------------------------------


async def test_the_bundle_leaves_out_program_variants_ignored_files_and_sealed_files(
    workspace: Any, tmp_path: Path
) -> None:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "task")
    (tmp_path / "ws" / "task" / ".agentic_task_ignore").write_text("secret.txt\n/skipped/\n")
    for name in (
        "secret.txt",
        "skipped/a.txt",
        "program_3/x.py",
        "__pycache__/x.pyc",
        ".agentic_runtime.json",
    ):
        path = tmp_path / "ws" / "task" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / "ws" / "task" / "keep.txt").write_text("kept")
    contract = await load_task_contract(workspace, "task")
    task = TaskRef(root="task", contract=contract, harness_evaluated=True)
    await write_program(workspace, "results/program_1", "def handle(request):\n    return 1.0\n")
    manifest = await extract_bundle(
        workspace,
        task=task,
        best_program_dir="results/program_1",
        manifest={"entrypoint_path": "src/program.py", "combined_score": 1.0, "correct": True},
        results_dir="results",
        reference_dir=None,
        output_dir="out",
        downstream_selection=None,
        evidence_dir=None,
        config=PosthocConfig(),
    )
    bundle = {entry.path for entry in await workspace.list_tree("out/_extracted_bundle")}
    assert "keep.txt" in bundle and "task.md" in bundle and "evaluate.py" in bundle
    assert "program_0/src/program.py" in bundle and "reference_program/program_spec.toml" in bundle
    for left_out in (
        "secret.txt",
        "skipped/a.txt",
        "program_3/x.py",
        "__pycache__/x.pyc",
        ".agentic_runtime.json",
        "evaluate_final.py",
        "task_contract.json",
        "resources/final/target.txt",
    ):
        assert left_out not in bundle, left_out
    assert "resources/search/target.txt" in bundle
    assert manifest["source_best_program_dir"] == "results/program_1"
    assert manifest["extraction_version"] == 1 and manifest["downstream_selection"] is None
    best = (await workspace.read_bytes("out/_extracted_bundle/program_0/src/program.py")).decode()
    assert "return 1.0" in best


async def test_extraction_refuses_missing_inputs(workspace: Any) -> None:
    from langgraph_agentic_scientist.posthoc.extract import ExtractionError

    task = await write_minimal_task(workspace)
    common: dict[str, Any] = {
        "task": task,
        "manifest": {},
        "results_dir": None,
        "reference_dir": None,
        "output_dir": "out",
        "downstream_selection": None,
        "evidence_dir": None,
        "config": PosthocConfig(),
    }
    with pytest.raises(ExtractionError, match="Best program directory does not exist"):
        await extract_bundle(workspace, best_program_dir="nowhere", **common)
    await workspace.write_bytes("noprog/src/program.py", b"x")
    with pytest.raises(ExtractionError, match=r"Invalid program_spec\.toml in best program"):
        await extract_bundle(workspace, best_program_dir="noprog", **common)
    await workspace.remove("task/program_0")
    with pytest.raises(ExtractionError, match="Task directory missing program_0"):
        await extract_bundle(workspace, best_program_dir="noprog", **common)
