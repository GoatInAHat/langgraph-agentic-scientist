"""The placeholders of the scaffold workflow texts that ``tools/derive_prompts.py`` extracts."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
EXTRACTED = THIRD_PARTY / "extracted" / "drafting_scaffold"


def test_the_placeholders_of_the_texts_are_the_documented_ones() -> None:
    expected = {
        "draft_retry_feedback.md": {"DIAGNOSTIC"},
        "continue_provisional_repair.md": {"DIAGNOSTIC_BASE64"},
        "alignment_report_correction.md": {"DIAGNOSTIC_BASE64"},
        "candidate_boundary_repair.md": {"DIAGNOSTIC_BASE64"},
        "qualification_repair.md": {
            "DIAGNOSTIC_BASE64",
            "DESIGN_BASE64",
            "ALIGNMENT_REPORT_BASE64",
            "CALIBRATION_RECORD_BASE64",
            "POST_VALIDATION_BASE64",
            "BOUNDARY_POLICY",
            "REPAIR_POLICY",
        },
        "task_gate.md": set(),
    }
    for name, names in expected.items():
        text = (EXTRACTED / name).read_text(encoding="utf-8")
        assert set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", text)) == names, name
