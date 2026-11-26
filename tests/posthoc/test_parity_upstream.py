"""The post-hoc rules equal the vendored upstream functions on the same inputs."""

from __future__ import annotations

import json
import os
import pickle
import random
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from langgraph_agentic_scientist.posthoc.common import (
    compare_metric_guardrail,
    find_checkpoint_candidates,
    infer_primary_metric,
    validate_checkpoint_loadable,
    validate_expected_outputs,
)
from langgraph_agentic_scientist.posthoc.models import (
    AnalysisCandidate,
    SelectionAnswer,
    is_valid_analysis_id,
    normalize_expected_outputs,
    normalize_implementation_plan,
    normalize_string_list,
)
from langgraph_agentic_scientist.posthoc.select import (
    apply_selection_constraints,
    decisions_from_answer,
    selection_payload,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace

Loader = Callable[..., ModuleType]
SELECT_STUBS = {
    "src.investigation.agent_prompt": {"stage_agent_prompt": None},
    "src.investigation.script_runners": {"_execute_script_command": None},
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


COMMON_STUBS = {
    "src.investigation.files": {"read_json": _read_json},
    "src.investigation.program_clone": {"clone_program_dir": None},
    "src.investigation.program_spec": {
        "ProgramSpecError": Exception,
        "load_program_spec": None,
        "resolve_entrypoint": None,
    },
}


@pytest.fixture
def models(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.posthoc_pipeline.models")


@pytest.fixture
def select(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.posthoc_pipeline.select", {**SELECT_STUBS, **COMMON_STUBS})


@pytest.fixture
def common(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.posthoc_pipeline.common", COMMON_STUBS)


# --- normalisation ------------------------------------------------------------------------------

LIST_VALUES: list[Any] = [
    None,
    "",
    "  ",
    " one ",
    "a\nb",
    ["a", " b ", "", "  ", 3, None, 2.5, True],
    [],
    ("x", "y"),
    5,
    0,
    {"k": "v"},
    b"bytes",
]

OUTPUT_VALUES: list[Any] = [
    ["results/table.csv", "plots/loss curve.png", "A summary of the results", "../escape.csv"],
    [
        "/abs/path.csv",
        "dir//double.csv",
        "dir/./dot.csv",
        "x" * 241 + ".csv",
        ("p" * 121) + "/a.csv",
    ],
    ["figure of accuracy", "Table 3", "result", "notes", "", "  ", "Ünïcode summary", "!!!"],
    ["file.CSV", "file.exe", "noext", "a b.csv", "a/b.json", "a\\b.csv", "-.txt", "."],
    "single.csv",
    "Bar chart of results",
    None,
    7,
    [1, 2, None],
    ["x" * 300, "boxplot_of_everything_" + "y" * 200],
]

PLAN_VALUES: list[Any] = [
    None,
    "",
    "  ",
    "1. load\n2. compute\n3. save",
    "- load\n* compute\n  \n- save",
    "1) load 2) compute 3) save",
    "1. only one",
    "  plain   sentence   spaced  ",
    ["step one", " step two ", "", "  "],
    list("abcdefghijklm"),
    list("1. load the data 2. fit 3. save the plot"),
    ["a", "b", "c", "dd", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
    ["a"] * 11 + ["bb"],
    [1, 2, 3],
    42,
    {"a": 1},
    "line1\n\nline2\r\nline3",
    "1. a 2. b\n",
    "10. ten 11. eleven 12. twelve",
]


@pytest.mark.parametrize("value", LIST_VALUES, ids=repr)
def test_string_lists_normalise_as_upstream_normalises_them(value: Any, models: ModuleType) -> None:
    assert normalize_string_list(value) == models.normalize_string_list(value)


@pytest.mark.parametrize("value", OUTPUT_VALUES, ids=repr)
def test_expected_outputs_normalise_as_upstream_normalises_them(
    value: Any, models: ModuleType
) -> None:
    assert normalize_expected_outputs(value) == models.normalize_expected_outputs(value)


@pytest.mark.parametrize("value", PLAN_VALUES, ids=repr)
def test_implementation_plans_normalise_as_upstream_normalises_them(
    value: Any, models: ModuleType
) -> None:
    assert normalize_implementation_plan(value) == models.normalize_implementation_plan(value)


@pytest.mark.parametrize(
    "value",
    ["a", "0", "a_b-c", "-a", "_a", "A", "a" * 128, "a" * 129, "", "a b", "a/b", "ä", "a\n", "a.b"],
)
def test_analysis_ids_are_validated_as_upstream_validates_them(
    value: str, models: ModuleType
) -> None:
    assert is_valid_analysis_id(value) == models.is_valid_analysis_id(value)


# --- selection ----------------------------------------------------------------------------------


def _port_candidate(analysis_id: str, **fields: Any) -> AnalysisCandidate:
    base: dict[str, Any] = {
        "id": analysis_id,
        "title": f"title {analysis_id}",
        "family": f"family {analysis_id}",
        "required_artifacts": [],
        "implementation_plan": ["inspect the saved weights"],
        "expected_outputs": ["out.csv"],
    }
    return AnalysisCandidate(**{**base, **fields})


def _upstream_candidate(models: ModuleType, port: AnalysisCandidate) -> Any:
    return models.AnalysisCandidate(
        analysis_id=port.analysis_id,
        title=port.title,
        family=port.family,
        hypothesis=port.hypothesis,
        relevance_reason=port.relevance_reason,
        feasibility_reason=port.feasibility_reason,
        required_artifacts=list(port.required_artifacts),
        requires_retrain=port.requires_retrain,
        implementation_plan=list(port.implementation_plan),
        expected_claim=port.expected_claim,
        expected_outputs=list(port.expected_outputs),
    )


def _populate_bundle(root: Path) -> None:
    files = {
        "task.md": "t",
        "program_0/results/metrics.json": "{}",
        "program_0/weights/model.pt": "w",
        "reference_program/notes.txt": "n",
        "extracted/data.csv": "d",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    os.symlink(root / "extracted", root / "linked_dir")
    os.symlink(root / "task.md", root / "program_0/link.md")
    os.symlink("/etc", root / "program_0/outside")


ARTIFACT_NAMES = [
    "task.md",
    "task_bundle/task.md",
    "results/metrics.json",
    "weights/model.pt",
    "program_0/weights/model.pt",
    "reference_program/notes.txt",
    "notes.txt",
    "extracted/data.csv",
    "linked_dir/data.csv",
    "link.md",
    "outside/hosts",
    "missing.txt",
    "../task.md",
    "/etc/hosts",
    "",
    "  ",
    "task_bundle/",
    "./task.md",
    "program_0/../task.md",
    "program_0",
    "weights",
]

PLANS = [
    ["inspect the saved weights"],
    ["Retrain the model with a new seed"],
    ["Train From Scratch"],
    ["fine-tune the head"],
    ["finetune it"],
    ["INVOKE EVALUATE.PY"],
    ["rerun evaluation", "plot"],
    ["run evaluation once"],
    ["plot the existing run", "retraining is not needed"],
    [],
]


@pytest.mark.parametrize("seed", range(40))
async def test_the_selection_constraints_equal_upstream_on_random_candidates(
    seed: int, select: ModuleType, models: ModuleType, tmp_path: Path
) -> None:
    rng = random.Random(seed)
    workspace_root = tmp_path / "ws"
    bundle = workspace_root / "run/ready/task_bundle"
    bundle.mkdir(parents=True)
    _populate_bundle(bundle)
    bootstrap_ready = rng.random() < 0.5
    (bundle.parent / "bootstrap_report.json").write_text(
        json.dumps({"bootstrap_ready": bootstrap_ready})
    )
    families = ["Heatmap", " heatmap ", "Ablation study", "layer-ABLATION", "curves", "tables", ""]
    candidates = [
        _port_candidate(
            f"a{index % 5}" if rng.random() < 0.25 else f"a{index}",
            family=rng.choice(families),
            title=rng.choice(["Same", " same ", "other", f"t{index}"]),
            requires_retrain=rng.random() < 0.15,
            implementation_plan=rng.choice(PLANS),
            required_artifacts=rng.sample(ARTIFACT_NAMES, rng.randint(0, 2)),
        )
        for index in range(rng.randint(0, 9))
    ]
    decisions = {
        candidate.analysis_id: (rng.random() < 0.8, f"why {candidate.analysis_id}")
        for candidate in candidates
        if rng.random() < 0.8
    }
    limit = rng.randint(0, 4)
    expected = select._apply_selection_constraints(
        candidates=[_upstream_candidate(models, c) for c in candidates],
        llm_decisions=decisions,
        max_selected_analyses=limit,
        bundle_dir=bundle,
    )
    actual = await apply_selection_constraints(
        LocalWorkspace(workspace_root),
        candidates=candidates,
        llm_decisions=decisions,
        max_selected_analyses=limit,
        bundle_dir="run/ready/task_bundle",
        bootstrap_ready=bootstrap_ready,
    )
    assert [c.analysis_id for c in actual.accepted] == [c.analysis_id for c in expected.accepted]
    assert [(r.candidate.analysis_id, r.reason) for r in actual.rejected] == [
        (c.analysis_id, reason) for c, reason in expected.rejected
    ]


@pytest.mark.parametrize("name", ARTIFACT_NAMES)
async def test_a_required_artifact_exists_as_upstream_finds_it(
    name: str, select: ModuleType, models: ModuleType, tmp_path: Path
) -> None:
    bundle = tmp_path / "ws/b"
    bundle.mkdir(parents=True)
    _populate_bundle(bundle)
    candidate = _port_candidate("a", required_artifacts=[name])
    expected = select._feasibility_error(
        candidate=_upstream_candidate(models, candidate), bundle_dir=bundle, bootstrap_ready=True
    )
    selection = await apply_selection_constraints(
        LocalWorkspace(tmp_path / "ws"),
        candidates=[candidate],
        llm_decisions={},
        max_selected_analyses=1,
        bundle_dir="b",
        bootstrap_ready=True,
    )
    actual = selection.rejected[0].reason if selection.rejected else None
    assert actual == expected


def test_the_selection_file_equals_upstream(
    select: ModuleType, models: ModuleType, tmp_path: Path
) -> None:
    candidates = [_port_candidate("a", hypothesis="h"), _port_candidate("b", requires_retrain=True)]
    upstream = models.AnalysisSelection(
        accepted=[_upstream_candidate(models, candidates[0])],
        rejected=[(_upstream_candidate(models, candidates[1]), "reason ü")],
    )
    select._write_selection_json(tmp_path, upstream, 3)
    from langgraph_agentic_scientist.posthoc.models import AnalysisSelection, RejectedCandidate

    port = AnalysisSelection(
        accepted=[candidates[0]],
        rejected=[RejectedCandidate(candidate=candidates[1], reason="reason ü")],
    )
    written = (tmp_path / "posthoc_selection.json").read_text(encoding="utf-8")
    assert json.dumps(selection_payload(port, 3), indent=2) == written


def test_the_selectors_decisions_equal_upstream_for_a_structured_answer(select: ModuleType) -> None:
    raw = {
        "decisions": [
            {"id": "a", "accepted": False, "reason": "no"},
            {"id": "b"},
            {"id": "", "accepted": False},
            {"id": "a", "accepted": True, "reason": "later wins"},
        ]
    }
    candidates = [_port_candidate(name) for name in ("a", "b", "c")]
    expected = select._parse_selection_decisions(raw)
    for candidate in candidates:
        expected.setdefault(candidate.analysis_id, (True, "accepted (not reviewed by selector)"))
    actual = decisions_from_answer(SelectionAnswer.model_validate(raw), candidates)
    assert {key: value for key, value in actual.items() if key} == {
        key: value for key, value in expected.items() if key
    }


@pytest.mark.parametrize(
    "raw",
    [
        [{"id": "a", "accepted": False}],
        {"accepted": [{"id": "a", "reason": "ok"}], "rejected": [{"analysis_id": "b"}]},
        {"decisions": "none"},
        "text",
        None,
    ],
    ids=repr,
)
def test_selector_shapes_the_structured_answer_cannot_take_are_the_declared_difference(
    raw: Any, select: ModuleType
) -> None:
    upstream = select._parse_selection_decisions(raw)
    if isinstance(raw, list) or (isinstance(raw, dict) and "accepted" in raw):
        assert upstream
    else:
        assert upstream is None
    try:
        answer = SelectionAnswer.model_validate(raw)
    except ValueError:
        return
    assert answer.decisions == []


# --- the primary metric and the guardrail -------------------------------------------------------

METRIC_CASES: list[dict[str, Any]] = [
    {"metrics": {"combined_score": 1.0, "val_accuracy": 0.9}},
    {"metrics": {"combined_score": 1.0, "accuracy": 0.8, "val_accuracy": 0.9}},
    {"metrics": {"combined_score": 1.0}},
    {"metrics": {"combined_score": 1.0, "weird_metric": 3}},
    {"metrics": {"correct": True, "combined_score": 2, "zeta": 5, "alpha": 6}},
    {"metrics": {"combined_score": 1.0, "train_loss": 0.5}, "results": {"auc": 0.7}},
    {"metrics": {}, "results": {"reward": 3, "combined_score": 1.5}},
    {"metrics": {}, "results": {}},
    {"metrics": None, "results": None},
    {"metrics": {"combined_score": "high"}},
    {"metrics": {"combined_score": float("nan")}},
    {"metrics": {"accuracy": float("nan"), "score": 1}},
    {"metrics": {"loss": 0.1, "perplexity": 2, "combined_score": 0.0}},
    {"metrics": {"combined_score": 1}, "requested_key": "combined_score"},
    {"metrics": {"combined_score": 1, "x": 2}, "requested_key": "x"},
    {"metrics": {"combined_score": 1, "x": 2}, "requested_key": "missing"},
    {"metrics": {"combined_score": 1}, "results": {"q": 3}, "requested_key": "q"},
    {"metrics": {"flag": True, "seed": 3, "time": 9}},
    {"metrics": {"flag": True, "combined_score": 4}},
    {"metrics": {"wall_time": 3, "val_error": 4}},
    {"metrics": {"Upper_Bound": 2, "lower_bound": 1}},
    {"metrics": {"x": 1}, "results": {"x": 2}},
    {"metrics": {"error": 1, "terminated_early": 0, "combined_score": 7}},
    {"metrics": {"lower_bound": 5, "combined_score": 1}},
]


@pytest.mark.parametrize("case", METRIC_CASES, ids=lambda c: json.dumps(c, default=str)[:90])
def test_the_primary_metric_is_inferred_as_upstream_infers_it(
    case: dict[str, Any], common: ModuleType
) -> None:
    try:
        expected: Any = common.infer_primary_metric(**case)
    except RuntimeError as error:
        expected = str(error)
    try:
        actual: Any = infer_primary_metric(**case)
    except RuntimeError as error:
        actual = str(error)
    assert _same(actual, expected)


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _info(key: str, value: float, higher: bool = True) -> dict[str, Any]:
    return {"key": key, "value": value, "source": "metrics.json", "higher_is_better": higher}


GUARDRAIL_CASES = [
    (_info("accuracy", 0.9), _info("accuracy", 0.9), 0.0),
    (_info("accuracy", 0.9), _info("accuracy", 0.89), 0.0),
    (_info("accuracy", 0.9), _info("accuracy", 0.89), 0.01),
    (_info("accuracy", 0.9), _info("accuracy", 0.885), 0.01),
    (_info("accuracy", 0.9), _info("accuracy", 1.2), 0.0),
    (_info("loss", 0.5, False), _info("loss", 0.5, False), 0.0),
    (_info("loss", 0.5, False), _info("loss", 0.51, False), 0.0),
    (_info("loss", 0.5, False), _info("loss", 0.51, False), 0.02),
    (_info("loss", 0.5, False), _info("loss", 0.4, False), 0.0),
    (_info("loss", 0.5, False), _info("accuracy", 0.4), 0.0),
    (_info("accuracy", 0.9), _info("accuracy", float("nan")), 0.1),
    (_info("accuracy", float("nan")), _info("accuracy", 0.9), 0.1),
    (_info("accuracy", float("inf")), _info("accuracy", float("inf")), 0.0),
    (_info("accuracy", 0.9), _info("accuracy", 0.9 - 1e-12), 0.0),
    (_info("accuracy", 0.9), _info("accuracy", 0.9), -0.5),
    ({**_info("accuracy", 0.9), "higher_is_better": 0}, _info("accuracy", 0.1), 0.0),
]


@pytest.mark.parametrize("baseline,candidate,tolerance", GUARDRAIL_CASES)
def test_the_guardrail_compares_as_upstream_compares(
    baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float, common: ModuleType
) -> None:
    expected = common.compare_metric_guardrail(
        baseline=baseline, candidate=candidate, tolerance=tolerance
    )
    actual = compare_metric_guardrail(baseline=baseline, candidate=candidate, tolerance=tolerance)
    assert _same(actual, expected)


# --- expected outputs ---------------------------------------------------------------------------

OUTPUT_PATHS = [
    "out.csv",
    "dir/out.csv",
    "dir/sub/deep.csv",
    "missing.csv",
    "dir",
    "dir/missing.csv",
    "link_file.csv",
    "link_dir/out.csv",
    "link_outside/hosts",
    "outside_link.csv",
    "/abs.csv",
    "/etc/hosts",
    "../up.csv",
    "dir/../out.csv",
    "./out.csv",
    "dir/./out.csv",
    "dir//out.csv",
    "",
    ".",
    "a b.csv",
    " out.csv",
    "out.csv ",
    "a\\b.csv",
    "~out.csv",
    "out.csv/",
]


def _populate_outputs(root: Path) -> None:
    for name, text in {
        "out.csv": "x",
        "dir/out.csv": "x",
        "dir/sub/deep.csv": "x",
        "a b.csv": "x",
        " out.csv": "x",
        "out.csv ": "x",
        "a\\b.csv": "x",
        "~out.csv": "x",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    os.symlink(root / "out.csv", root / "link_file.csv")
    os.symlink(root / "dir", root / "link_dir")
    os.symlink("/etc", root / "link_outside")
    os.symlink("/etc/hosts", root / "outside_link.csv")


async def _expected_outputs_pair(
    common: ModuleType, tmp_path: Path, names: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = tmp_path / "ws/analysis"
    root.mkdir(parents=True)
    _populate_outputs(root)
    expected = common.validate_expected_outputs(root, names)
    actual = await validate_expected_outputs(LocalWorkspace(tmp_path / "ws"), "analysis", names)
    return expected, actual


async def test_expected_outputs_are_validated_as_upstream_validates_them(
    common: ModuleType, tmp_path: Path
) -> None:
    names = [
        name
        for name in OUTPUT_PATHS
        if "\\" not in name and name.strip() == name and not name.startswith("~")
    ]
    names = [name for name in names if ".." not in name.split("/") and name not in {"", "."}]
    expected, actual = await _expected_outputs_pair(common, tmp_path, names)
    assert actual == expected
    assert "out.csv" in actual["found"] and "link_file.csv" in actual["missing"]


@pytest.mark.parametrize("name", ["dir/../out.csv", "a\\b.csv", " out.csv", "out.csv ", "~out.csv"])
async def test_the_workspace_path_rules_are_stricter_than_upstream_for_odd_names(
    name: str, common: ModuleType, tmp_path: Path
) -> None:
    expected, actual = await _expected_outputs_pair(common, tmp_path, [name])
    assert expected["found"] == [name]
    assert actual["missing"] == [name]
    assert normalize_expected_outputs([name]) != [name]


@pytest.mark.parametrize("name", ["", ".", "../up.csv", "/abs.csv", "/etc/hosts"])
async def test_empty_and_escaping_expected_outputs_are_missing_in_both(
    name: str, common: ModuleType, tmp_path: Path
) -> None:
    expected, actual = await _expected_outputs_pair(common, tmp_path, [name])
    assert actual == expected == {"found": [], "missing": [name], "all_present": False}


# --- checkpoints --------------------------------------------------------------------------------


def _write_checkpoint_tree(root: Path) -> None:
    files: dict[str, bytes] = {
        "_results.json": json.dumps(
            {
                "checkpoint_path": "weights/best.pt",
                "model": "model_final.pkl",
                "model_path": "/abs/model.pt",
                "other_path": "../escape.pt",
                "checkpoint": 7,
                "dir_path": "weights",
                "missing_path": "nope.pt",
                "alias_path": "alias.pt",
                "notes": "model_final.pkl",
            }
        ).encode(),
        "weights/best.pt": b"x",
        "weights/real.pt": b"x",
        "model_final.pkl": b"x",
        "checkpoints/a.bin": b"x",
        "ckpt_dir/plain.npz": b"x",
        "plain.npz": b"x",
        "state.safetensors": b"x",
        "best_train.pt": b"x",
        "test_model.pt": b"x",
        "model.txt": b"x",
        "MODEL.PT": b"x",
        ".hidden/model.pt": b"x",
        "a/weights/anything.joblib": b"x",
        "data_model.pkl": b"x",
        "snapshot.ckpt": b"x",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    os.symlink(root / "weights", root / "link_dir")
    os.symlink(root / "weights/best.pt", root / "alias.pt")
    os.symlink(root / "weights/best.pt", root / "model_link.pt")


@pytest.fixture
def neutral(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("w")


async def test_checkpoint_candidates_equal_upstream(common: ModuleType, neutral: Path) -> None:
    tmp_path = neutral
    root = tmp_path / "ws/p"
    root.mkdir(parents=True)
    _write_checkpoint_tree(root)
    expected = [
        path.relative_to(root.resolve()).as_posix()
        for path in common.find_checkpoint_candidates(root)
    ]
    actual = [
        name for name, _ in await find_checkpoint_candidates(LocalWorkspace(tmp_path / "ws"), "p")
    ]
    assert actual == expected
    assert "weights/best.pt" in actual and "model_link.pt" not in actual


async def test_upstream_reads_the_name_hints_in_the_absolute_path_and_the_port_in_the_relative_one(
    common: ModuleType, neutral: Path
) -> None:
    program = neutral / "ws/checkpoints/p"
    program.mkdir(parents=True)
    (program / "plain.npz").write_bytes(b"x")
    (program / "model.npz").write_bytes(b"x")
    expected = [
        path.relative_to(program.resolve()).as_posix()
        for path in common.find_checkpoint_candidates(program)
    ]
    actual = [
        name
        for name, _ in await find_checkpoint_candidates(
            LocalWorkspace(neutral / "ws"), "checkpoints/p"
        )
    ]
    assert expected == ["model.npz", "plain.npz"]
    assert actual == ["model.npz"]


async def test_checkpoint_candidates_of_a_missing_or_empty_program_equal_upstream(
    common: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "ws/empty").mkdir(parents=True)
    workspace = LocalWorkspace(tmp_path / "ws")
    for name in ("empty", "missing"):
        assert common.find_checkpoint_candidates(tmp_path / "ws" / name) == []
        assert await find_checkpoint_candidates(workspace, name) == []


async def test_a_results_path_through_a_linked_directory_is_found_only_by_the_name_scan(
    common: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "ws/p"
    (root / "latest_target").mkdir(parents=True)
    (root / "latest_target/final.pt").write_bytes(b"x")
    os.symlink(root / "latest_target", root / "latest")
    (root / "_results.json").write_text(json.dumps({"checkpoint_path": "latest/final.pt"}))
    expected = [
        path.relative_to(root.resolve()).as_posix()
        for path in common.find_checkpoint_candidates(root)
    ]
    actual = [
        name for name, _ in await find_checkpoint_candidates(LocalWorkspace(tmp_path / "ws"), "p")
    ]
    assert expected == ["latest_target/final.pt"]
    assert actual == []


def _npz() -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.npy", b"x")
    return buffer.getvalue()


def _empty_zip() -> bytes:
    import io

    buffer = io.BytesIO()
    zipfile.ZipFile(buffer, "w").close()
    return buffer.getvalue()


def _safetensors(header: bytes, declared: int | None = None) -> bytes:
    length = len(header) if declared is None else declared
    return struct.pack("<Q", length) + header


CHECKPOINT_BYTES: dict[str, tuple[str, bytes]] = {
    "pickle_ok.pkl": ("pkl", pickle.dumps({"a": [1, 2]}, protocol=2)),
    "pickle_joblib.joblib": ("joblib", pickle.dumps([1], protocol=0)),
    "pickle_truncated.pkl": ("pkl", pickle.dumps({"a": [1, 2]}, protocol=2)[:-3]),
    "pickle_garbage.pickle": ("pickle", b"not a pickle at all"),
    "torch_pickle.pt": ("pt", pickle.dumps({"w": 1}, protocol=2)),
    "torch_garbage.pth": ("pth", b"\x00\x01garbage"),
    "torch_zip.ckpt": ("ckpt", _npz()),
    "torch_empty_zip.bin": ("bin", _empty_zip()),
    "npz_ok.npz": ("npz", _npz()),
    "npz_bad.npz": ("npz", b"nope"),
    "st_ok.safetensors": ("safetensors", _safetensors(b'{"a": 1}')),
    "st_short.safetensors": ("safetensors", b"abc"),
    "st_tiny_header.safetensors": ("safetensors", _safetensors(b"{", 1)),
    "st_overlong.safetensors": ("safetensors", _safetensors(b"{}", 99)),
    "st_not_json.safetensors": ("safetensors", _safetensors(b"not json!")),
    "st_not_utf8.safetensors": ("safetensors", _safetensors(b"\xff\xfe\xfd")),
    "json_ok.json": ("json", b'{"a": 1}'),
    "json_bad.json": ("json", b"{"),
    "unknown.weights": ("weights", b"anything"),
    "empty.pt": ("pt", b""),
}


@pytest.mark.parametrize("name", sorted(CHECKPOINT_BYTES))
async def test_a_checkpoint_is_validated_as_upstream_validates_it(
    name: str, common: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    (root / "c").mkdir(parents=True)
    (root / "c" / name).write_bytes(CHECKPOINT_BYTES[name][1])
    expected = common.validate_checkpoint_loadable(root / "c" / name)
    actual = await validate_checkpoint_loadable(LocalWorkspace(root), f"c/{name}")
    assert actual[0] == expected[0]
    if expected[1] is None or actual[1] is None or "not loadable" not in expected[1]:
        assert actual[1] == expected[1]
    else:
        assert actual[1] is not None and "not loadable" in actual[1]


async def test_a_missing_symlinked_or_directory_checkpoint_equals_upstream(
    common: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    (root / "c/dir").mkdir(parents=True)
    (root / "c/real.pkl").write_bytes(pickle.dumps(1))
    os.symlink(root / "c/real.pkl", root / "c/link.pkl")
    workspace = LocalWorkspace(root)
    for name in ("missing.pkl", "dir", "link.pkl", "real.pkl"):
        expected = common.validate_checkpoint_loadable(root / "c" / name)
        assert await validate_checkpoint_loadable(workspace, f"c/{name}") == expected


async def test_an_npy_checkpoint_is_validated_by_its_header_without_numpy(
    common: ModuleType, tmp_path: Path
) -> None:
    try:
        import numpy as np
    except ImportError:
        np = None
    root = tmp_path / "ws"
    (root / "c").mkdir(parents=True)
    (root / "c/bad.npy").write_bytes(b"not numpy")
    expected = common.validate_checkpoint_loadable(root / "c/bad.npy")
    actual = await validate_checkpoint_loadable(LocalWorkspace(root), "c/bad.npy")
    assert expected[0] is False and actual[0] is False
    if np is not None:
        import io

        buffer = io.BytesIO()
        np.save(buffer, np.arange(4))
        (root / "c/ok.npy").write_bytes(buffer.getvalue())
        assert await validate_checkpoint_loadable(LocalWorkspace(root), "c/ok.npy") == (True, None)
        assert common.validate_checkpoint_loadable(root / "c/ok.npy") == (True, None)
