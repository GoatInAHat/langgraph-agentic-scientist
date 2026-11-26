"""Idea bundles: the ideator's structured answer, its validation and its harness rules.

Ports of upstream ``orchestrator/idea_bundle.py`` (normalisation), ``investigation/
idea_generation.py`` (``load_idea_artifacts``: the cross-checks and the harness rules) and
``idea_retry_feedback.py`` (the bounded feedback of a retry). The ideator answers through
structured output (:class:`IdeaDraft`), so what upstream read from two JSON files and repaired by
hand arrives as one typed object; every rule that validated the files applies to the draft.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ConfigDict, Field

from ..core.state import IdeaBundle, RubricSpec, StateModel
from ..prompts import PromptRegistry, render_text
from .rubrics import rubric_metric_check

NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
MULTI_UNDERSCORE_RE = re.compile(r"_+")
MAX_SOURCE_PATTERNS = 8
INCUMBENT_GATE_ID = "incumbent_score_improvement"
VALIDATION_ERROR_PREFIX = "Invalid idea artifacts: "


class IdeaBundleValidationError(RuntimeError):
    """The idea bundle is missing required structure (upstream ``IdeaBundleValidationError``)."""


class IdeaArtifactsError(ValueError):
    """A generated idea is invalid or inconsistent (upstream ``IdeaArtifactsError``).

    Attributes:
        reason: Upstream's reason code, for example ``invalid_idea_bundle``.
        detail: Human-readable detail, fed back to the ideator on a retry.
    """

    def __init__(self, reason: str, *, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason)


class _Draft(StateModel):
    """Base of the ideator's structured answer: a state record that ignores unknown fields.

    Being a :class:`StateModel`, the draft is covered by ``serde_allowlist()`` (it is stored in the
    ideator session's checkpoints); unknown fields are ignored, as upstream's normalisation
    ignores keys it does not read.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class RubricVerificationDraft(_Draft):
    """The ``verification`` of an implementation rubric: a static source check."""

    type: str = Field("source_contains", description="Always `source_contains`.")
    patterns: list[str] = Field(
        description="1 to 8 specific literal snippets the implementation will contain."
    )


class RubricDraft(_Draft):
    """One target rubric as the ideator proposes it."""

    id: str = Field("", description="Stable snake_case rubric id.")
    description: str = Field(description="Concrete, measurable description of the rubric.")
    type: str = Field("implementation", description="`metric` or `implementation`.")
    verification: RubricVerificationDraft | None = Field(
        None, description="For implementation rubrics of harness-evaluated attempts only."
    )


class IdeaDraft(_Draft):
    """The ideator's structured answer: upstream's ``idea_bundle.json``."""

    idea_id: str = Field("", description="A stable id for the idea.")
    title: str = Field(description="A short title.")
    hypothesis: str = Field(description="The hypothesis of this cycle's idea.")
    implementation_plan: list[str] = Field(
        description="A non-empty list of short implementation steps."
    )
    target_rubrics: list[RubricDraft] = Field(
        description="A non-empty list of rubrics: at least one metric and one implementation."
    )
    notes_for_agent: str = Field("", description="Notes for the implementing agent.")


def slugify(text: str, *, fallback: str = "rubric") -> str:
    """Lowercase ``text`` and join its alphanumeric runs with underscores."""
    clean = NON_ALNUM_RE.sub("_", text.strip().lower())
    clean = MULTI_UNDERSCORE_RE.sub("_", clean).strip("_")
    return clean or fallback


def without_target_rubric_verification(payload: Any) -> Any:
    """Return a copy of an idea payload without the optional ``verification`` of its rubrics."""

    def strip_rubrics(raw_rubrics: list[Any]) -> list[Any]:
        stripped: list[Any] = []
        for rubric in raw_rubrics:
            if not isinstance(rubric, dict):
                stripped.append(rubric)
                continue
            normalized = dict(rubric)
            normalized.pop("verification", None)
            stripped.append(normalized)
        return stripped

    if isinstance(payload, list):
        return strip_rubrics(payload)
    if isinstance(payload, dict) and isinstance(payload.get("target_rubrics"), list):
        normalized_payload = dict(payload)
        normalized_payload["target_rubrics"] = strip_rubrics(payload["target_rubrics"])
        return normalized_payload
    return payload


def _normalize_source_contains(raw: Any, index: int, max_chars: int) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("type") != "source_contains":
        raise IdeaBundleValidationError(
            f"target_rubrics[{index}].verification must use type source_contains"
        )
    patterns = raw.get("patterns")
    if not isinstance(patterns, list) or not 1 <= len(patterns) <= MAX_SOURCE_PATTERNS:
        raise IdeaBundleValidationError(
            f"target_rubrics[{index}].verification.patterns must contain 1 to "
            f"{MAX_SOURCE_PATTERNS} strings"
        )
    normalized: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > max_chars:
            raise IdeaBundleValidationError(
                f"target_rubrics[{index}].verification.patterns must be non-empty strings "
                f"up to {max_chars} characters"
            )
        normalized.append(pattern.strip())
    return {"type": "source_contains", "patterns": normalized}


def normalize_target_rubrics(raw: Any, *, max_pattern_chars: int = 1024) -> list[dict[str, Any]]:
    """Normalise target rubrics: ids slugified and de-duplicated, verification checked."""
    if not isinstance(raw, list):
        raise IdeaBundleValidationError("target_rubrics must be a list")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IdeaBundleValidationError(f"target_rubrics[{idx}] must be an object")
        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise IdeaBundleValidationError(
                f"target_rubrics[{idx}].description must be non-empty string"
            )
        rubric_type = item.get("type")
        if not isinstance(rubric_type, str) or not rubric_type.strip():
            rubric_type = "implementation"
        rubric_id = item.get("id")
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            rubric_id = slugify(desc)
        rubric_id = slugify(rubric_id)
        dedupe = rubric_id
        suffix = 2
        while dedupe in seen_ids:
            dedupe = f"{rubric_id}_{suffix}"
            suffix += 1
        seen_ids.add(dedupe)
        rubric: dict[str, Any] = {
            "id": dedupe,
            "description": desc.strip(),
            "type": rubric_type.strip().lower(),
        }
        verification = item.get("verification")
        if verification is not None:
            if rubric["type"] != "implementation":
                raise IdeaBundleValidationError(
                    f"target_rubrics[{idx}].verification is only supported for implementation "
                    "rubrics"
                )
            rubric["verification"] = _normalize_source_contains(
                verification, idx, max_pattern_chars
            )
        normalized.append(rubric)
    if not normalized:
        raise IdeaBundleValidationError("target_rubrics must not be empty")
    return normalized


def normalize_idea_bundle(
    raw: Any, cycle_index: int, *, max_pattern_chars: int = 1024
) -> dict[str, Any]:
    """Normalise an idea bundle (upstream ``normalize_idea_bundle``)."""
    if not isinstance(raw, dict):
        raise IdeaBundleValidationError("idea bundle must be a JSON object")
    idea_id = raw.get("idea_id")
    if not isinstance(idea_id, str) or not idea_id.strip():
        idea_id = f"idea_{cycle_index:04d}"
    idea_id = slugify(idea_id, fallback=f"idea_{cycle_index:04d}")
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise IdeaBundleValidationError("title must be non-empty string")
    hypothesis = raw.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise IdeaBundleValidationError("hypothesis must be non-empty string")
    plan = raw.get("implementation_plan")
    if not isinstance(plan, list) or not plan:
        raise IdeaBundleValidationError("implementation_plan must be a non-empty list")
    plan_items = [str(item).strip() for item in plan if str(item).strip()]
    if not plan_items:
        raise IdeaBundleValidationError("implementation_plan must contain non-empty items")
    target_rubrics = normalize_target_rubrics(
        raw.get("target_rubrics"), max_pattern_chars=max_pattern_chars
    )
    notes = raw.get("notes_for_agent")
    if not isinstance(notes, str):
        notes = ""
    return {
        "idea_id": idea_id,
        "title": title.strip(),
        "hypothesis": hypothesis.strip(),
        "implementation_plan": plan_items,
        "target_rubrics": target_rubrics,
        "notes_for_agent": notes.strip(),
    }


def correct_incumbent_score(best: dict[str, Any] | None, *, harness: bool) -> float | None:
    """Return a score gate only for a scientifically correct incumbent (harness mode only)."""
    if not harness or not best:
        return None
    score = best.get("combined_score")
    if (
        best.get("correct") is not True
        or isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(float(score))
    ):
        return None
    return float(score)


def _replace_metric_rubrics(
    rubrics: list[dict[str, Any]], incumbent_score: float | None
) -> list[dict[str, Any]]:
    implementation = [rubric for rubric in rubrics if rubric.get("type") == "implementation"]
    if incumbent_score is None:
        return implementation
    if isinstance(incumbent_score, bool) or not math.isfinite(float(incumbent_score)):
        raise IdeaArtifactsError(
            "invalid_harness_incumbent_score",
            detail="harness incumbent combined_score must be finite",
        )
    threshold = float(incumbent_score)
    gate = {
        "id": INCUMBENT_GATE_ID,
        "description": f"combined_score > {threshold!r}",
        "type": "metric",
        "metric": "combined_score",
        "operator": ">",
        "threshold": threshold,
    }
    return [gate, *implementation]


def validate_idea(
    draft: IdeaDraft | dict[str, Any],
    cycle_index: int,
    *,
    harness: bool,
    incumbent_score: float | None,
    max_harness_rubrics: int = 32,
    max_pattern_chars: int = 1024,
) -> IdeaBundle:
    """Validate the ideator's answer as upstream validates the two idea files.

    Outside harness mode the optional ``verification`` is dropped. In harness mode every
    agent-authored metric rubric is replaced by the canonical ``incumbent_score_improvement``
    gate (``combined_score > incumbent``; none without a correct incumbent), at least one
    implementation rubric is required, and each implementation rubric needs a
    ``source_contains`` verification.

    Raises:
        IdeaArtifactsError: With upstream's reason code and a detail fed back on a retry.
    """
    raw: Any = draft.model_dump(exclude_none=True) if isinstance(draft, IdeaDraft) else draft
    if not harness:
        raw = without_target_rubric_verification(raw)
    try:
        bundle = normalize_idea_bundle(raw, cycle_index, max_pattern_chars=max_pattern_chars)
    except IdeaBundleValidationError as exc:
        raise IdeaArtifactsError("invalid_idea_bundle", detail=str(exc)) from exc
    rubrics: list[dict[str, Any]] = bundle["target_rubrics"]
    if harness:
        rubrics = _replace_metric_rubrics(rubrics, incumbent_score)
        if not any(rubric.get("type") == "implementation" for rubric in rubrics):
            raise IdeaArtifactsError(
                "missing_harness_implementation_rubric",
                detail="harness mode requires at least one implementation rubric",
            )
        if len(rubrics) > max_harness_rubrics:
            raise IdeaArtifactsError(
                "too_many_harness_rubrics",
                detail=f"harness mode supports at most {max_harness_rubrics} rubrics",
            )
        for rubric in rubrics:
            rubric_type = rubric.get("type")
            if rubric_type == "metric":
                if rubric_metric_check(rubric) is None:
                    raise IdeaArtifactsError(
                        "missing_harness_metric_threshold",
                        detail=(
                            f"metric rubric {rubric.get('id')!r} requires a numeric metric "
                            "threshold using one direct evaluator-emitted metric key"
                        ),
                    )
                continue
            if rubric_type != "implementation":
                continue
            verification = rubric.get("verification")
            if not isinstance(verification, dict) or verification.get("type") != "source_contains":
                raise IdeaArtifactsError(
                    "missing_harness_implementation_verification",
                    detail=(
                        f"implementation rubric {rubric.get('id')!r} requires source_contains "
                        "verification"
                    ),
                )
    bundle["target_rubrics"] = rubrics
    return IdeaBundle(
        idea_id=bundle["idea_id"],
        title=bundle["title"],
        hypothesis=bundle["hypothesis"],
        implementation_plan=bundle["implementation_plan"],
        target_rubrics=[RubricSpec.model_validate(rubric) for rubric in rubrics],
        notes_for_agent=bundle["notes_for_agent"],
    )


def rubric_json(rubric: RubricSpec) -> dict[str, Any]:
    """Return a rubric as upstream writes it into ``target_rubrics.json``.

    The incumbent gate carries its structured comparison (``metric``, ``operator``,
    ``threshold``) next to its description, as upstream's gate does.
    """
    data: dict[str, Any] = {"id": rubric.id, "description": rubric.description, "type": rubric.type}
    if rubric.verification is not None:
        data["verification"] = rubric.verification
    for key in ("metric", "operator", "threshold"):
        value = getattr(rubric, key)
        if value is not None:
            data[key] = value
    return data


def bundle_json(bundle: IdeaBundle) -> dict[str, Any]:
    """Return a bundle as upstream writes it into ``idea_bundle.json``."""
    return {
        "idea_id": bundle.idea_id,
        "title": bundle.title,
        "hypothesis": bundle.hypothesis,
        "implementation_plan": list(bundle.implementation_plan),
        "target_rubrics": [rubric_json(rubric) for rubric in bundle.target_rubrics],
        "notes_for_agent": bundle.notes_for_agent,
    }


def dump_json(value: Any) -> str:
    """Serialise as upstream writes its idea files (indented, sorted keys)."""
    return json.dumps(value, indent=2, sort_keys=True)


def _slug_component(text: str | None, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    if slug:
        return slug
    return re.sub(r"[^a-z0-9]+", "_", fallback.strip().lower()).strip("_") or "idea"


def idea_dir_name(*, cycle_index: int, title: str | None, idea_id: str) -> str:
    """Return ``<cycle:02d>_<slug of the title>`` (upstream ``state.idea_dir_name``)."""
    return f"{cycle_index:02d}_{_slug_component(title, idea_id)}"


def _printable(text: str) -> str:
    out: list[str] = []
    for character in text.encode("utf-8", errors="backslashreplace").decode("utf-8"):
        codepoint = ord(character)
        if character.isprintable():
            out.append(character)
        elif codepoint <= 0xFF:
            out.append(f"\\x{codepoint:02x}")
        elif codepoint <= 0xFFFF:
            out.append(f"\\u{codepoint:04x}")
        else:
            out.append(f"\\U{codepoint:08x}")
    return "".join(out)


def sanitize_validation_feedback(diagnostic: str, *, max_bytes: int = 2048) -> str:
    """Escape control characters and cut a diagnostic at ``max_bytes`` (with ``...``)."""
    safe = _printable(diagnostic)
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    return encoded[: max_bytes - 3].decode("utf-8", errors="ignore") + "..."


def render_validation_feedback(
    prompts: PromptRegistry, diagnostic: str, *, max_bytes: int = 2048
) -> str:
    """Return the text appended to the ideator's prompt on a retry (upstream wording).

    The diagnostic is untrusted data and goes in as a JSON object, never as an instruction.
    """
    if not diagnostic:
        return ""
    head = render_text(prompts.text("workflow/investigation/idea_validation_feedback"), {})
    return head + json.dumps(
        {"diagnostic": sanitize_validation_feedback(diagnostic, max_bytes=max_bytes)},
        ensure_ascii=False,
    )


__all__ = [
    "INCUMBENT_GATE_ID",
    "VALIDATION_ERROR_PREFIX",
    "IdeaArtifactsError",
    "IdeaBundleValidationError",
    "IdeaDraft",
    "RubricDraft",
    "RubricVerificationDraft",
    "bundle_json",
    "correct_incumbent_score",
    "dump_json",
    "idea_dir_name",
    "normalize_idea_bundle",
    "normalize_target_rubrics",
    "render_validation_feedback",
    "rubric_json",
    "sanitize_validation_feedback",
    "slugify",
    "validate_idea",
    "without_target_rubric_verification",
]
