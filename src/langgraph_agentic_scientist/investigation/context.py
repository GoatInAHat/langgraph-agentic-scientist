"""The placeholder tables and context blocks of the idea and attempt prompts.

Upstream's drivers (``idea.sh``, ``run.sh``) fill the prompt and the staged skills from the
results directory: recent ideas, recent attempts, recent memory, the best manifest and the
distilled conclusions, each in a fixed format and count. The formats and counts are package data
(``workflow/investigation/context_blocks.toml``); this module reads the files the agents and the
host keep and renders the blocks the same way.
"""

from __future__ import annotations

import json
import math
import posixpath
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.contract import TaskContract, indented_public_search_payload
from ..core.errors import TaskContractError
from ..core.state import TaskRef
from ..prompts import PromptRegistry, render_text
from ..workspace.paths import join
from ..workspace.protocol import Workspace

OUTER_STATE_FILE = "outer_state.json"
IDEAS_HISTORY_FILE = "ideas_history.jsonl"
ATTEMPTS_HISTORY_FILE = "attempts_history.jsonl"
IDEA_GENERATION_FAILURES_FILE = "idea_generation_failures.jsonl"
BEST_MANIFEST_FILE = "best_program_manifest.json"
GLOBAL_MEMORY_MD_FILE = "memory.md"
GLOBAL_MEMORY_JSONL_FILE = "memory.jsonl"
CONCLUSIONS_FILE = "conclusions.md"
OTHER_WORKERS_CONCLUSIONS_FILE = "conclusions_other_workers.md"
CONCLUSIONS_HISTORY_FILE = "conclusions_history.jsonl"
MEMORY_HEADER = "# Global Memory Log\n\n"
SHARED_DIR = "_shared_memory"
"""Directory of an attempt's workspace that holds its view of the shared memory files."""

BLOCKS_KEY = "workflow/investigation/context_blocks.toml"
FEEDBACK_KEY = "workflow/investigation/harness_feedback.toml"
MAX_SEARCH_FEEDBACK_BYTES = 8 * 1024


def load_blocks(prompts: PromptRegistry) -> dict[str, Any]:
    """Parse the context-block formats of the registry (upstream's formats, as package data)."""
    return tomllib.loads(prompts.text(BLOCKS_KEY))


def display_path(path: str, base: str) -> str:
    """Render ``path`` relative to ``base`` for prompt text (upstream ``relative_display_path``).

    Both are workspace-relative POSIX paths; the result is ``.``, ``./child`` or ``../..``.
    """
    relative = posixpath.relpath(path or ".", base or ".")
    if relative == ".":
        return "."
    if relative.startswith("../"):
        return relative
    return f"./{relative}"


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def summarize_jsonl_entries(
    entries: Sequence[Any], keys: Sequence[str], blocks: Mapping[str, Any]
) -> str:
    """Summarise the last entries of a JSONL file as upstream's ``summarize_jsonl_entries``."""
    spec = blocks["jsonl_summary"]
    if not entries:
        return str(spec["empty"])
    lines: list[str] = []
    shown = int(blocks["counts"]["items_shown"])
    for index, entry in enumerate(list(entries)[-shown:], start=1):
        if not isinstance(entry, dict):
            lines.append(render_text(spec["non_object"], {"INDEX": str(index)}))
            continue
        parts: list[str] = []
        for key in keys:
            value = entry.get(key)
            if _is_empty(value):
                continue
            if isinstance(value, dict | list):
                value = json.dumps(value, sort_keys=True)
            parts.append(render_text(spec["part"], {"KEY": key, "VALUE": str(value)}))
        if not parts:
            parts.append(str(spec["no_fields"]))
        lines.append(
            render_text(
                spec["item"], {"INDEX": str(index), "PARTS": str(spec["separator"]).join(parts)}
            )
        )
    return "\n".join(lines)


def summarize_best_manifest(manifest: Mapping[str, Any] | None, blocks: Mapping[str, Any]) -> str:
    """Summarise ``best_program_manifest.json`` as upstream's ``summarize_best_manifest``."""
    spec = blocks["best_manifest"]
    if not manifest:
        return str(spec["none"])
    parts: list[str] = []
    for key in spec["keys"]:
        value = manifest.get(key)
        if _is_empty(value):
            continue
        parts.append(render_text(spec["part"], {"KEY": key, "VALUE": str(value)}))
    if not parts:
        return str(spec["empty"])
    return str(spec["prefix"]) + str(spec["separator"]).join(parts)


def search_metric_feedback(
    contract: TaskContract | None,
    metrics: Mapping[str, Any],
    *,
    max_bytes: int = MAX_SEARCH_FEEDBACK_BYTES,
) -> dict[str, int | float]:
    """Project the declared search scalars of ``metrics`` (upstream ``search_metric_feedback``).

    Without a contract nothing is declared and nothing is projected. Values are finite numbers
    (booleans excluded); the JSON of the projection stays within ``max_bytes``.
    """
    if contract is None:
        return {}
    projected: dict[str, int | float] = {}
    for metric in contract.metrics.search:
        value = metrics.get(metric.name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        try:
            if not math.isfinite(value):
                continue
        except OverflowError:
            continue
        proposed = {**projected, metric.name: value}
        if len(json.dumps(proposed, ensure_ascii=True).encode("utf-8")) <= max_bytes:
            projected = proposed
    return projected


def summarize_recent_attempts(
    contract: TaskContract | None,
    entries: Sequence[Any],
    blocks: Mapping[str, Any],
    *,
    max_bytes: int = MAX_SEARCH_FEEDBACK_BYTES,
) -> str:
    """Render recent attempts as bounded data (upstream ``summarize_recent_attempts``)."""
    spec = blocks["recent_attempts"]
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary: dict[str, Any] = {key: entry[key] for key in spec["keys"] if key in entry}
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            scientific: dict[str, Any] = dict(
                search_metric_feedback(contract, metrics, max_bytes=max_bytes)
            )
            score = metrics.get("combined_score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                try:
                    if math.isfinite(score):
                        scientific["combined_score"] = score
                except OverflowError:
                    pass
            if isinstance(metrics.get("correct"), bool):
                scientific["correct"] = metrics["correct"]
            summary["metrics"] = scientific
        encoded = json.dumps(summary, ensure_ascii=True, sort_keys=True)
        if len(encoded.encode()) > max_bytes + 1024:
            encoded = json.dumps({"metrics": summary.get("metrics", {})}, ensure_ascii=True)
        lines.append(str(spec["prefix"]) + encoded)
    return "\n".join(lines) if lines else str(spec["empty"])


def jsonl_records(text: str) -> list[dict[str, Any]]:
    """Parse the JSON-object lines of a JSONL text; other lines are skipped."""
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


async def read_text_or(workspace: Workspace, path: str, default: str = "") -> str:
    """Read a UTF-8 file of the workspace, or return ``default`` when it is absent."""
    try:
        return (await workspace.read_bytes(path)).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return default


async def tail_records(workspace: Workspace, path: str, max_lines: int) -> list[dict[str, Any]]:
    """Return the JSON-object records among the last ``max_lines`` lines of a JSONL file."""
    lines = (await read_text_or(workspace, path)).splitlines()[-max_lines:]
    return jsonl_records("\n".join(lines))


async def conclusions_body(workspace: Workspace, path: str, fallback: str) -> str:
    """Return a conclusions file's stripped body, or ``fallback`` when absent or empty."""
    body = (await read_text_or(workspace, path)).strip()
    return body or fallback


async def read_json_object(workspace: Workspace, path: str) -> dict[str, Any] | None:
    """Read a JSON object file; ``None`` when absent, unparsable or not an object."""
    text = await read_text_or(workspace, path)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def public_search_contract_section(
    prompts: PromptRegistry, contract: TaskContract | None, *, max_bytes: int = 16 * 1024
) -> str:
    """Return the Public Search Contract section of the candidate author's prompt.

    Empty for a task without a contract. The section frames the candidate-visible subset of the
    contract as task data (upstream ``public_search_contract_section``).

    Raises:
        TaskContractError: When the section exceeds ``max_bytes``.
    """
    if contract is None:
        return ""
    rendered = render_text(
        prompts.text("workflow/investigation/public_search_contract"),
        {"INDENTED_PAYLOAD": indented_public_search_payload(contract)},
    )
    if len(rendered.encode("utf-8")) > max_bytes:
        raise TaskContractError("public search contract exceeds prompt limit")
    return "\n\n" + rendered


def harness_rubric_requirements(
    prompts: PromptRegistry, *, harness: bool, max_pattern_chars: int
) -> str:
    """Return upstream's ``HARNESS_RUBRIC_REQUIREMENTS`` text for the mode."""
    key = "workflow/investigation/harness_rubric_requirements_" + ("on" if harness else "off")
    return render_text(prompts.text(key), {"MAX_SOURCE_PATTERN_CHARS": str(max_pattern_chars)})


async def idea_values(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    results_dir: str,
    staging_dir: str,
    cycle_index: int,
    worker_index: int,
    harness: bool,
    contract: TaskContract | None,
    max_pattern_chars: int,
) -> dict[str, str]:
    """Return the placeholder values of ``idea.md`` and its skills (``idea.sh:653-693``).

    Paths are rendered relative to the ideator's root, the worker's results root
    ``results_dir``; ``staging_dir`` is the fresh directory of this generation attempt.
    """
    blocks = load_blocks(prompts)
    counts = blocks["counts"]
    keys = blocks["jsonl_summary"]
    recent_ideas = await tail_records(
        workspace, join(results_dir, IDEAS_HISTORY_FILE), int(counts["recent_ideas"])
    )
    recent_attempts = await tail_records(
        workspace, join(results_dir, ATTEMPTS_HISTORY_FILE), int(counts["recent_attempts"])
    )
    memory_recent = await tail_records(
        workspace, join(results_dir, GLOBAL_MEMORY_JSONL_FILE), int(counts["idea_memory_lines"])
    )
    own = await conclusions_body(
        workspace,
        join(results_dir, CONCLUSIONS_FILE),
        str(blocks["conclusions"]["own_fallback"]),
    )
    others = await conclusions_body(
        workspace, join(results_dir, OTHER_WORKERS_CONCLUSIONS_FILE), ""
    )
    placeholder = str(blocks["conclusions"]["other_workers_placeholder"]).strip()
    others_section = (
        render_text(str(blocks["conclusions"]["other_workers_section"]), {"BODY": others})
        if others and others != placeholder
        else ""
    )
    best = await read_json_object(workspace, join(results_dir, BEST_MANIFEST_FILE))

    def shown(name: str) -> str:
        return display_path(join(results_dir, name), results_dir)

    return {
        "CYCLE_INDEX": str(cycle_index),
        "WORKER_INDEX": str(worker_index),
        "IDEA_OUTPUT_DIR": display_path(staging_dir, results_dir),
        "IDEA_BUNDLE_FILE": display_path(join(staging_dir, "idea_bundle.json"), results_dir),
        "TARGET_RUBRICS_FILE": display_path(join(staging_dir, "target_rubrics.json"), results_dir),
        "GLOBAL_RESULTS_DIR": display_path(results_dir, results_dir),
        "IDEAS_HISTORY_FILE": shown(IDEAS_HISTORY_FILE),
        "ATTEMPTS_HISTORY_FILE": shown(ATTEMPTS_HISTORY_FILE),
        "BEST_MANIFEST_FILE": shown(BEST_MANIFEST_FILE),
        "RECENT_IDEAS_SUMMARY": summarize_jsonl_entries(recent_ideas, keys["idea_keys"], blocks),
        "RECENT_ATTEMPTS_SUMMARY": summarize_recent_attempts(contract, recent_attempts, blocks),
        "GLOBAL_MEMORY_MD_FILE": shown(GLOBAL_MEMORY_MD_FILE),
        "GLOBAL_MEMORY_JSONL_FILE": shown(GLOBAL_MEMORY_JSONL_FILE),
        "GLOBAL_MEMORY_RECENT_SUMMARY": summarize_jsonl_entries(
            memory_recent, keys["memory_keys"], blocks
        ),
        "BEST_MANIFEST_SUMMARY": summarize_best_manifest(best, blocks),
        "CONCLUSIONS_MD_FILE": shown(CONCLUSIONS_FILE),
        "CONCLUSIONS_HISTORY_FILE": shown(CONCLUSIONS_HISTORY_FILE),
        "OWN_CONCLUSIONS": own,
        "OTHER_WORKERS_CONCLUSIONS_SECTION": others_section,
        "HARNESS_RUBRIC_REQUIREMENTS": harness_rubric_requirements(
            prompts, harness=harness, max_pattern_chars=max_pattern_chars
        ),
    }


async def attempt_values(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    task: TaskRef,
    experiment_dir: str,
    shared_dir: str,
    attempt_id: str,
    idea_id: str,
    idea_dir_name: str,
    max_program_evals: int,
    worker_index: int,
    cycle_index: int,
    active_idea: str,
    active_rubrics: str,
    include_public_contract: bool,
    public_contract_max_bytes: int,
) -> dict[str, str]:
    """Return the placeholder values of ``general.md``/``harness_edit.md`` and their skills.

    Upstream renders the attempt prompt from inside the experiment directory, so ``TASK_DIR`` and
    ``RESULTS_DIR`` are ``.``. The memory and conclusions files named by the prompt are the
    attempt's view of the shared files under ``shared_dir`` (inside the experiment directory).
    """
    blocks = load_blocks(prompts)
    keys = blocks["jsonl_summary"]
    memory_recent = await tail_records(
        workspace,
        join(shared_dir, GLOBAL_MEMORY_JSONL_FILE),
        int(blocks["counts"]["attempt_memory_lines"]),
    )
    own = await conclusions_body(
        workspace,
        join(shared_dir, CONCLUSIONS_FILE),
        str(blocks["conclusions"]["own_fallback"]),
    )
    others = await conclusions_body(
        workspace,
        join(shared_dir, OTHER_WORKERS_CONCLUSIONS_FILE),
        str(blocks["conclusions"]["other_workers_fallback"]),
    )

    def shown(name: str) -> str:
        return display_path(join(shared_dir, name), experiment_dir)

    return {
        "ACTIVE_IDEA_JSON": active_idea,
        "ACTIVE_RUBRICS_JSON": active_rubrics,
        "ACTIVE_IDEA_FILE": "./active_idea.json",
        "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
        "ATTEMPT_ID": attempt_id,
        "IDEA_ID": idea_id,
        "IDEA_DIR_NAME": idea_dir_name,
        "ATTEMPT_DIR_NAME": "idea_root",
        "MAX_PROGRAM_EVALS_PER_ATTEMPT": str(max_program_evals),
        "WORKER_INDEX": str(worker_index),
        "CYCLE_INDEX": str(cycle_index),
        "PUBLIC_SEARCH_CONTRACT_SECTION": (
            public_search_contract_section(
                prompts, task.contract, max_bytes=public_contract_max_bytes
            )
            if include_public_contract
            else ""
        ),
        "GLOBAL_MEMORY_MD_FILE": shown(GLOBAL_MEMORY_MD_FILE),
        "GLOBAL_MEMORY_JSONL_FILE": shown(GLOBAL_MEMORY_JSONL_FILE),
        "GLOBAL_MEMORY_RECENT_SUMMARY": summarize_jsonl_entries(
            memory_recent, keys["memory_keys"], blocks
        ),
        "CONCLUSIONS_MD_FILE": shown(CONCLUSIONS_FILE),
        "CONCLUSIONS_HISTORY_FILE": shown(CONCLUSIONS_HISTORY_FILE),
        "OWN_CONCLUSIONS": own,
        "OTHER_WORKERS_CONCLUSIONS": others,
    }


__all__ = [
    "ATTEMPTS_HISTORY_FILE",
    "BEST_MANIFEST_FILE",
    "BLOCKS_KEY",
    "CONCLUSIONS_FILE",
    "CONCLUSIONS_HISTORY_FILE",
    "FEEDBACK_KEY",
    "GLOBAL_MEMORY_JSONL_FILE",
    "GLOBAL_MEMORY_MD_FILE",
    "IDEAS_HISTORY_FILE",
    "IDEA_GENERATION_FAILURES_FILE",
    "MEMORY_HEADER",
    "OTHER_WORKERS_CONCLUSIONS_FILE",
    "OUTER_STATE_FILE",
    "SHARED_DIR",
    "attempt_values",
    "conclusions_body",
    "display_path",
    "harness_rubric_requirements",
    "idea_values",
    "jsonl_records",
    "load_blocks",
    "public_search_contract_section",
    "read_json_object",
    "read_text_or",
    "search_metric_feedback",
    "summarize_best_manifest",
    "summarize_jsonl_entries",
    "summarize_recent_attempts",
    "tail_records",
]
