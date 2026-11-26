"""Candidate screening, selection (fail-open, hard caps) and custom baselines."""

from __future__ import annotations

import pytest

from langgraph_agentic_scientist.variants.custom import (
    inject_custom_baselines,
    load_custom_baselines,
)
from langgraph_agentic_scientist.variants.generate import compute_program_diff, screen_candidates
from langgraph_agentic_scientist.variants.models import (
    VariantBundleAnswer,
    VariantCandidate,
    VariantDecision,
    VariantDecisionsAnswer,
    VariantProposal,
    VariantSelection,
)
from langgraph_agentic_scientist.variants.select import (
    apply_hard_caps,
    decisions_from_answer,
    fail_open_decisions,
    selection_json,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace


def _candidate(variant_id: str, kind: str = "ablation", **fields: str) -> VariantCandidate:
    return VariantCandidate(variant_id=variant_id, title=variant_id, kind=kind, **fields)


def _proposal(variant_id: str, kind: str = "ablation") -> VariantProposal:
    return VariantProposal(variant_id=variant_id, title="T", kind=kind)  # type: ignore[arg-type]


def test_screening_drops_malformed_reserved_and_duplicate_ids() -> None:
    answer = VariantBundleAnswer(
        decision="propose_variants",
        candidates=[
            _proposal("ablation_ok"),
            _proposal("Bad-Id"),
            _proposal("control_best"),
            _proposal("baseline_reference", "baseline"),
            _proposal("ablation_ok"),
            _proposal("x" * 80),
            _proposal("baseline_simple", "baseline"),
        ],
    )
    screened = screen_candidates(answer)
    assert [c.variant_id for c in screened.candidates] == ["ablation_ok", "baseline_simple"]
    assert {reason for _, reason in screened.dropped} == {
        "malformed variant id",
        "reserved variant id",
        "duplicate variant id",
    }


def test_no_relevant_variants_gives_no_candidates_whatever_the_list_holds() -> None:
    answer = VariantBundleAnswer(
        decision="no_relevant_variants", candidates=[_proposal("ablation_ok")]
    )
    assert screen_candidates(answer).candidates == []
    assert VariantBundleAnswer().decision == "no_relevant_variants"  # upstream's default


def test_the_reviewer_is_fail_open_and_unmentioned_candidates_are_accepted() -> None:
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    assert fail_open_decisions(candidates, "accepted (agent failed)") == {
        "a": (True, "accepted (agent failed)"),
        "b": (True, "accepted (agent failed)"),
        "c": (True, "accepted (agent failed)"),
    }
    answer = VariantDecisionsAnswer(
        decisions=[VariantDecision(variant_id="a", accepted=False, reason="tuning sweep")]
    )
    decisions = decisions_from_answer(candidates, answer)
    assert decisions["a"] == (False, "tuning sweep")
    assert decisions["b"] == (True, "accepted (not evaluated by agent)")
    # an unknown id in the answer changes nothing
    stray = VariantDecisionsAnswer(decisions=[VariantDecision(variant_id="zzz", accepted=False)])
    stray_decisions = decisions_from_answer(candidates, stray)
    assert all(stray_decisions[c.variant_id][0] for c in candidates)


def test_hard_caps_apply_in_candidate_order() -> None:
    candidates = [_candidate(f"ablation_{i}") for i in range(5)] + [
        _candidate(f"baseline_{i}", "baseline") for i in range(4)
    ]
    decisions = {c.variant_id: (True, "ok") for c in candidates}
    decisions["ablation_1"] = (False, "multi-change")
    selection = apply_hard_caps(candidates, decisions, max_ablations=3, max_extra_baselines=3)
    assert [c.variant_id for c in selection.accepted] == [
        "ablation_0",
        "ablation_2",
        "ablation_3",
        "baseline_0",
        "baseline_1",
        "baseline_2",
    ]
    reasons = {r.candidate.variant_id: r.reason for r in selection.rejected}
    assert reasons == {
        "ablation_1": "multi-change",
        "ablation_4": "ablation cap reached (3)",
        "baseline_3": "baseline cap reached (3)",
    }
    assert selection.controls == ["control_best"]
    written = selection_json(selection)
    assert written["num_accepted"] == 6 and written["num_rejected"] == 3
    assert {entry["source"] for entry in written["accepted"]} == {"generated"}  # type: ignore[union-attr]


def test_zero_accepted_is_valid() -> None:
    candidates = [_candidate("a")]
    selection = apply_hard_caps(
        candidates, {"a": (False, "filler")}, max_ablations=3, max_extra_baselines=3
    )
    assert selection.accepted == [] and len(selection.rejected) == 1


# --- custom baselines ----------------------------------------------------------------------------


async def test_markdown_custom_baseline(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes(
        "custom/dfa.md",
        (
            b"---\nvariant_id: dfa\ntitle: Direct Feedback Alignment\nhypothesis: random feedback\n"
            b"target_components: [a, b]\n---\n\n## Description\nBody text.\n"
        ),
    )
    (candidate,) = await load_custom_baselines(workspace, ["custom/dfa.md"])
    assert candidate.variant_id == "baseline_dfa"  # renamed with the prefix, as upstream does
    assert candidate.title == "Direct Feedback Alignment"
    assert candidate.kind == "baseline" and candidate.base_program == "best"
    assert candidate.target_components == ["a", "b"]
    assert candidate.description_body == "## Description\nBody text."
    assert candidate.reference_dir is None


async def test_folder_custom_baseline_with_overrides(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("custom/My Method-v2/README.md", b"Reference readme.\n")
    await workspace.write_bytes("custom/My Method-v2/code.py", b"x = 1\n")
    await workspace.write_bytes(
        "custom/other/baseline.md",
        b"---\nvariant_id: baseline_x\ntitle: X\nhypothesis: h\n---\nOverride.\n",
    )
    folder, override = await load_custom_baselines(
        workspace, ["custom/My Method-v2", "custom/other"]
    )
    assert folder.variant_id == "baseline_my_method_v2"
    assert folder.title == "My Method V2" and folder.base_program == "reference"
    assert folder.description_body == "Reference readme."
    assert folder.reference_dir == "custom/My Method-v2"
    assert override.variant_id == "baseline_x" and override.hypothesis == "h"
    assert override.description_body == "Override."
    assert override.base_program == "reference"  # a folder baseline starts from reference code


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("no frontmatter", "does not contain YAML frontmatter"),
        ("---\nvariant_id: a\n", "no closing"),
        ("---\n- list\n---\n", "not a mapping"),
        ("---\ntitle: t\n---\n", "Missing required 'variant_id'"),
        ("---\nvariant_id: Bad-Id\n---\n", "Invalid variant_id"),
        ("---\nvariant_id: a\nkind: control\n---\n", "Invalid kind"),
    ],
)
async def test_custom_baseline_errors(
    workspace: LocalWorkspace, content: str, message: str
) -> None:
    await workspace.write_bytes("c.md", content.encode())
    with pytest.raises(ValueError, match=message):
        await load_custom_baselines(workspace, ["c.md"])


async def test_custom_baseline_missing_and_duplicate(workspace: LocalWorkspace) -> None:
    with pytest.raises(FileNotFoundError):
        await load_custom_baselines(workspace, ["nowhere.md"])
    await workspace.write_bytes("a.md", b"---\nvariant_id: baseline_a\n---\n")
    await workspace.write_bytes("b.md", b"---\nvariant_id: baseline_a\n---\n")
    with pytest.raises(ValueError, match="Duplicate variant_id"):
        await load_custom_baselines(workspace, ["a.md", "b.md"])


def test_injection_replaces_same_id_and_custom_only_discards_generated() -> None:
    generated = VariantSelection(
        accepted=[_candidate("baseline_a", "baseline", hypothesis="generated"), _candidate("abl")]
    )
    custom = [
        _candidate("baseline_a", "baseline", description_body="custom"),
        _candidate("baseline_b", "baseline", description_body="x"),
    ]
    merged = inject_custom_baselines(generated, custom, custom_only=False)
    assert [(c.variant_id, c.description_body) for c in merged.accepted] == [
        ("abl", ""),
        ("baseline_a", "custom"),
        ("baseline_b", "x"),
    ]
    only = inject_custom_baselines(generated, custom, custom_only=True)
    assert [c.variant_id for c in only.accepted] == ["baseline_a", "baseline_b"]
    assert only.controls == []
    assert inject_custom_baselines(None, custom, custom_only=False).controls == ["control_best"]


# --- the program diff ----------------------------------------------------------------------------


async def test_program_diff_cases(workspace: LocalWorkspace) -> None:
    assert await compute_program_diff(workspace, "b", max_chars=100) == (
        "(no reference program available for diff)"
    )
    await workspace.write_bytes("b/reference_program/src/p.py", b"a = 1\n")
    await workspace.write_bytes("b/program_0/src/p.py", b"a = 1\n")
    assert await compute_program_diff(workspace, "b", max_chars=100) == "(programs are identical)"
    await workspace.write_bytes("b/program_0/src/p.py", b"a = 2\n" + b"# " + b"x" * 300 + b"\n")
    diff = await compute_program_diff(workspace, "b", max_chars=100)
    assert diff.endswith("\n... (diff truncated)") and len(diff) == 100 + len(
        "\n... (diff truncated)"
    )
    assert diff.startswith("diff -ruN reference_program/src/p.py program_0/src/p.py")
    full = await compute_program_diff(workspace, "b", max_chars=50_000)
    assert "-a = 1" in full and "+a = 2" in full
