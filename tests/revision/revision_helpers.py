"""Helpers of the revision tests: a published paper, a revision of it and a stub paper graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from langgraph_agentic_scientist.core.results import PaperResult
from langgraph_agentic_scientist.review.models import ReviewerAnswer
from langgraph_agentic_scientist.testing import structured_call

PAPER = "paper"
MANUSCRIPT = "A study of circle packing and search, revised for the reviewers. " * 8
REVISION_ROOT = "revision"


def make_pdf(text: str) -> bytes:
    """A one-page PDF with ``text`` (needs PyMuPDF)."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=11)
    data = bytes(document.tobytes())
    document.close()
    return data


def sha(data: bytes | str) -> str:
    """The hex SHA-256 of ``data``."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def review_dict(**overrides: Any) -> dict[str, Any]:
    """A valid review under upstream's keys."""
    review: dict[str, Any] = {
        "Summary": "The paper studies packing.",
        "Strengths": ["Clear method."],
        "Weaknesses": ["Small evaluation.", "Claims are broad."],
        "Originality": 3,
        "Quality": 3,
        "Clarity": 3,
        "Significance": 2,
        "Questions": ["Why this benchmark?"],
        "Limitations": ["Single task."],
        "Ethical Concerns": False,
        "Soundness": 3,
        "Presentation": 3,
        "Contribution": 2,
        "Overall": 4,
        "Confidence": 4,
        "Decision": "Reject",
    }
    for key, value in overrides.items():
        review[key.replace("_", " ")] = value
    return review


def answer(review: dict[str, Any] | None = None) -> AIMessage:
    """The scripted structured answer of one reviewer call."""
    return structured_call(ReviewerAnswer, {"thought": "t", "review": review or review_dict()})


def put(root: Path, relative: str, data: bytes | str | dict[str, Any]) -> None:
    """Write a file below ``root`` (a dict is written as JSON)."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def read_json(root: Path, relative: str) -> dict[str, Any]:
    """Read a JSON object file below ``root``."""
    return json.loads((root / relative).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def paper_data(*, revised: bool = False, **changes: Any) -> dict[str, Any]:
    """The ``paper_data.json`` of the original, or of a revision that keeps everything."""
    data: dict[str, Any] = {
        "generated_at": "2026-01-01T00:00:00Z" if not revised else "2026-02-02T00:00:00Z",
        "results_dir": "/runs/a/results" if not revised else "/runs/b/results",
        "all_programs": [
            {"id": "program_0", "score": 0.5, "path": "/runs/a/p0"},
            {"id": "program_1", "score": 0.7, "path": "/runs/a/p1"},
        ],
        "grounding_sources": {"result.best": {"value": 0.7}},
        "final_evaluation": {"outcome": "qualified", "metrics": {"score": 0.69}},
        "posthoc_analyses": [
            {"id": "a1", "table": [1, 2], "key_finding_for_paper": "original wording"}
        ],
        "downstream_study": {"selection": "program_1"},
    }
    if revised:
        data["posthoc_analyses"] = [
            {"id": "a1", "table": [1, 2], "key_finding_for_paper": "new wording", "extra": 1}
        ]
        data["all_programs"].append({"id": "program_2", "score": 0.1, "path": "/runs/b/p2"})
    data.update(changes)
    return data


def audit_files(inputs: dict[str, str]) -> dict[str, Any]:
    """The audit record and the validation report that records it, for audit inputs ``inputs``."""
    audit = {
        "verdict": "pass",
        "summary": "no findings",
        "finding_count": 0,
        "findings": [],
        "input_sha256": inputs,
    }
    report = {
        "ok": True,
        "agentic_evidence_audit": {
            "checked": True,
            "verdict": "pass",
            "finding_count": 0,
            "findings": [],
            "input_sha256": inputs,
        },
    }
    return {"paper_evidence_audit.json": audit, "validation_report.json": report}


def write_original(root: Path, paper: str = PAPER) -> dict[str, bytes]:
    """Write a published, reviewed paper; return the bytes of the files that a revision replaces."""
    files: dict[str, bytes] = {}

    def add(name: str, data: bytes | str | dict[str, Any]) -> None:
        put(root, f"{paper}/{name}", data)
        files[name] = (root / paper / name).read_bytes()

    inputs = {
        "evidence-source/paper_data.json": sha("data"),
        "tex-source/paper.tex": sha("tex"),
        "evidence-source/old_note.txt": sha("old note"),
        "evidence-tree/sources": sha("tree"),
        "evidence-source/latest_manifest.json": sha("completion"),
        "evidence-source/ai_review.before_revision.json": sha("review"),
    }
    add("paper.tex", "\\documentclass{article} original")
    add("paper.pdf", make_pdf("The original manuscript. " * 10))
    add("paper_data.json", paper_data())
    add("claims_grounding.json", {"claims": ["c1"]})
    add("references.bib", "@article{a, title={A}}")
    add("references_provenance.json", {"a": {"source": "s2"}})
    add("figure_manifest.json", {"figures": ["f1"]})
    add("method_scope_audit.json", {"shared_method_elements": ["x"]})
    add("method_scope_audit.md", "scope")
    for name, payload in audit_files(inputs).items():
        add(name, payload)
    add("old_note.txt", "old note")
    add("sources/a.txt", "tree file a")
    add("sources/deep/b.txt", "tree file b")
    add("figures/f1.png", b"png-original")
    add(
        "latest_manifest.json",
        {
            "paper_pdf": f"{paper}/paper.pdf",
            "ai_review": {"status": "completed", "review_json": f"{paper}/ai_review.json"},
            "before_revision": {
                "status": "completed",
                "review_json": f"{paper}/ai_review.before_revision.json",
                "log": f"{paper}/ai_review.before_revision.log",
            },
        },
    )
    add("ai_review.before_revision.json", review_dict())
    add("ai_review.before_revision.log", "")
    add("ai_review.json", review_dict())
    return files


def write_revision(
    root: Path,
    directory: str,
    *,
    paper: str = PAPER,
    data: dict[str, Any] | None = None,
    bound: dict[str, str] | None = None,
    audit_override: dict[str, Any] | None = None,
    report_override: dict[str, Any] | None = None,
    skip: tuple[str, ...] = (),
    manifest: dict[str, Any] | None = None,
) -> None:
    """Write a revision of the paper of :func:`write_original` into ``directory``."""
    inputs = bound or {
        "evidence-source/paper_data.json": sha("data"),
        "tex-source/paper.tex": sha("tex2"),
        "evidence-source/new_note.txt": sha("new note"),
        "evidence-source/latest_manifest.json": sha("completion2"),
    }
    files: dict[str, bytes | str | dict[str, Any]] = {
        "paper.tex": "\\documentclass{article} revised",
        "paper.pdf": make_pdf(MANUSCRIPT),
        "paper_data.json": data if data is not None else paper_data(revised=True),
        "claims_grounding.json": {"claims": ["c1", "c2"]},
        "references.bib": "@article{a, title={A}}",
        "references_provenance.json": {"a": {"source": "s2"}},
        "figure_manifest.json": {"figures": ["f1", "f2"]},
        "method_scope_audit.json": {"shared_method_elements": ["x"]},
        "method_scope_audit.md": "scope revised",
        "new_note.txt": "new note",
        "figures/f1.png": b"png-revised",
        "figures/f2.png": b"png-new",
        "latest_manifest.json": manifest
        if manifest is not None
        else {
            "paper_pdf": f"{directory}/paper.pdf",
            "figures": [f"{directory}/figures/f1.png", f"{directory}/figures/f2.png"],
            "volatile": "revision-only text",
        },
    }
    for name, payload in audit_files(inputs).items():
        files[name] = payload
    if audit_override is not None:
        files["paper_evidence_audit.json"] = {
            **files["paper_evidence_audit.json"],
            **audit_override,
        }  # type: ignore[dict-item]
    if report_override is not None:
        files["validation_report.json"] = {**files["validation_report.json"], **report_override}  # type: ignore[dict-item]
    for name, payload in files.items():
        if name not in skip:
            put(root, f"{directory}/{name}", payload)


class StubPaper:
    """A stand-in for the paper graph in revision mode; it writes a revision and records calls."""

    def __init__(
        self,
        root: Path,
        *,
        result: Callable[[str], PaperResult] | None = None,
        error: BaseException | None = None,
        **revision: Any,
    ) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.revision = revision
        self.result = result
        self.error = error

    async def __call__(self, payload: dict[str, Any], runtime: Any) -> PaperResult:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        directory = payload["output_dir"]
        if self.result is not None:
            return self.result(directory)
        write_revision(self.root, directory, **self.revision)
        return PaperResult(
            status="complete",
            paper_dir=directory,
            pdf_path=f"{directory}/paper.pdf",
            tex_path=f"{directory}/paper.tex",
            manifest_path=f"{directory}/latest_manifest.json",
        )


def tree(root: Path, relative: str) -> dict[str, bytes]:
    """Every file below ``root/relative`` as ``{relative path: bytes}``."""
    base = root / relative
    return {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }
