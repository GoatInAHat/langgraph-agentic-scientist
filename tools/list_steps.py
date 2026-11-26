#!/usr/bin/env python3
"""Generate ``docs/STEPS.md`` and the diagrams of ``docs/ARCHITECTURE.md`` from the compiled graphs.

Every graph factory of the package is built with its defaults and walked: each step that carries
hook nodes (``<phase>.<step>.before`` and ``.after``) is listed with the state keys a hook may
update, and each graph is drawn with ``get_graph(xray=1)`` as a Mermaid diagram. Nothing in the
output is written by hand, so the listing cannot fall out of date.

Usage::

    uv run python tools/list_steps.py            # write both files
    uv run python tools/list_steps.py --check    # fail when either file is out of date (CI)

A factory that cannot be built (its subpackage is absent) is reported on standard error, listed as
unavailable, and makes ``--check`` fail.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS_PATH = REPO_ROOT / "docs" / "STEPS.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "ARCHITECTURE.md"
BEGIN = "<!-- BEGIN GENERATED GRAPHS (tools/list_steps.py) -->"
END = "<!-- END GENERATED GRAPHS -->"


@dataclass(frozen=True)
class GraphEntry:
    """One exported graph factory."""

    title: str
    factory: str
    summary: str


GRAPHS: tuple[GraphEntry, ...] = (
    GraphEntry("pipeline", "build_pipeline_graph", "the end-to-end pipeline"),
    GraphEntry("task drafting", "build_task_drafting_graph", "task drafting, A2-A7"),
    GraphEntry("task design", "build_task_design_graph", "the design loop, A1"),
    GraphEntry("investigation", "build_investigation_graph", "the investigation campaign"),
    GraphEntry("campaign worker", "build_campaign_worker_graph", "one campaign worker"),
    GraphEntry("ideation", "build_ideation_graph", "one cycle's idea generation"),
    GraphEntry("attempt", "build_attempt_graph", "one cycle's attempt"),
    GraphEntry(
        "downstream selection", "build_downstream_selection_graph", "the no-winner fallback"
    ),
    GraphEntry("variants", "build_variants_graph", "ablations and baselines"),
    GraphEntry("final evaluation", "build_final_evaluation_graph", "the sealed final evaluation"),
    GraphEntry("post-hoc", "build_posthoc_graph", "post-hoc analysis"),
    GraphEntry("paper", "build_paper_graph", "paper writing"),
    GraphEntry("references", "build_references_graph", "reference planning and retrieval"),
    GraphEntry("review", "build_review_graph", "the AI review"),
    GraphEntry("revision", "build_revision_graph", "the revision"),
)

SLOTS: dict[str, str] = {
    "investigation.ideation": "one cycle's idea generation (the ideation graph)",
    "investigation.attempt": "one cycle's attempt (the attempt graph)",
    "investigation.worker": "one campaign worker (the worker graph)",
    "variants.run_variant": "the run of one variant (edit, evaluate, verify)",
    "variants.cross_task": "the cross-task study",
    "final_evaluation.draw_inputs": "drawing the sealed final inputs after the candidate is frozen",
    "posthoc.analysis": "the run of one post-hoc analysis",
    "paper.data": "building the paper data from the phase results",
    "paper.references": "reference planning and retrieval",
    "paper.postdraft_enrichment": "post-draft reference enrichment",
    "paper.validation": "the validation gate",
    "paper.audit": "the evidence audit",
    "paper.review_gate": "the in-loop review gate",
    "paper.figures": "concept figure preparation",
    "review": "the AI review that the revision runs after its edit",
    "revision.paper": "the paper graph in revision mode",
    "revision.audit_input_hashes": "the input hashes of the revision's evidence audit",
    "revision.scope_audit": "the revision's method scope audit",
}
"""Every slot key a factory reads from ``GraphOverrides.slots`` (whole phases are replaced
through ``GraphOverrides.phases`` instead)."""


def _all_graphs(graph: Any) -> list[Any]:
    """Return a compiled graph and every compiled graph mounted below it."""
    found = [graph]
    for _name, sub in graph.get_subgraphs(recurse=True):
        found.append(sub)
    return found


def _writable(node: Any) -> tuple[str, ...] | None:
    """Return the state keys a hook node lets a hook update (its closure's ``writable``)."""
    bound = getattr(node, "bound", None)
    function = getattr(bound, "afunc", None) or getattr(bound, "func", None)
    if function is None:
        return None
    try:
        writable = inspect.getclosurevars(function).nonlocals.get("writable")
    except (TypeError, ValueError):
        return None
    return tuple(sorted(writable)) if writable is not None else None


def hook_keys(graph: Any) -> dict[str, tuple[str, ...]]:
    """Return every hook key of a graph and its mounted graphs, with its hook-writable keys."""
    keys: dict[str, tuple[str, ...]] = {}
    for compiled in _all_graphs(graph):
        for name, node in compiled.nodes.items():
            if name.endswith(".before"):
                keys[name.removesuffix(".before")] = _writable(node) or ()
    return dict(sorted(keys.items()))


def _build(entry: GraphEntry, warnings: list[str]) -> Any | None:
    import langgraph_agentic_scientist as package
    from langgraph_agentic_scientist import GraphOverrides
    from langgraph_agentic_scientist import pipeline as pipeline_module

    try:
        factory = getattr(package, entry.factory)
        if entry.factory != "build_pipeline_graph":
            return factory()
    except Exception as exc:  # a phase that cannot be built is reported, not fatal
        warnings.append(f"{entry.factory}: not available ({type(exc).__name__}: {exc})")
        return None
    placeholders = {}
    for spec in pipeline_module.PHASE_SPECS:
        try:
            module = importlib.import_module(f"langgraph_agentic_scientist.{spec.module}")
            getattr(module, spec.factory)(overrides=GraphOverrides(), name=spec.name)
        except Exception as exc:
            warnings.append(
                f"pipeline: {spec.factory} not available; a placeholder is mounted "
                f"({type(exc).__name__}: {exc})"
            )
            placeholders[spec.name] = _placeholder(spec.name)
    return factory(overrides=GraphOverrides(phases=placeholders))


def _placeholder(name: str) -> Any:
    from langgraph.graph import END, START, StateGraph

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(dict)
    builder.add_node("placeholder", lambda state: {})
    builder.add_edge(START, "placeholder")
    builder.add_edge("placeholder", END)
    return builder.compile(name=name)


def render_steps(built: Sequence[tuple[GraphEntry, Any | None]]) -> str:
    """Render ``docs/STEPS.md``."""
    lines = [
        "# Steps, hook keys and slots",
        "",
        "Generated by `tools/list_steps.py` from the compiled graphs; do not edit by hand.",
        "",
        "Every step of the method has two hook nodes, `<graph>.<step>.before` and",
        "`<graph>.<step>.after`. A hook is registered under the step's key, for example",
        '`Hooks(before={"pipeline.paper": [fn]})`. It receives the state and the runtime and may',
        'return `{"hook_outputs": {...}}`, which is merged under the step\'s key, and updates of',
        "the state keys listed for the step (a step that lists none accepts `hook_outputs` only);",
        "any other key raises `HookError`. A completed hook is checkpointed and does not run again",
        "after a crash or a resume.",
        "",
        "Slots are replaceable steps: `GraphOverrides(slots={key: graph_or_callable})` replaces",
        "the step at compile time; the hooks of the step that mounts a slot still run around it.",
        "",
    ]
    all_keys: set[str] = set()
    for entry, graph in built:
        lines += [f"## {entry.title} (`{entry.factory}`)", ""]
        if graph is None:
            lines += ["Not available in this build.", ""]
            continue
        keys = hook_keys(graph)
        all_keys |= set(keys)
        lines += [f"{entry.summary[0].upper()}{entry.summary[1:]}.", ""]
        lines += ["| Hook key | Hook-writable state keys |", "|---|---|"]
        for key, writable in keys.items():
            cell = ", ".join(f"`{item}`" for item in writable) if writable else "none"
            lines.append(f"| `{key}` | {cell} |")
        lines.append("")
    lines += ["## Slots", "", "| Slot key | Replaces |", "|---|---|"]
    for key, meaning in SLOTS.items():
        lines.append(f"| `{key}` | {meaning} |")
    lines.append("")
    return "\n".join(lines)


def _diagram(graph: Any) -> str:
    """Draw ``graph`` without the step metadata that every node of a ``StepGraph`` carries."""
    from langgraph_agentic_scientist.core.steps import PHASE_METADATA_KEY, STEP_METADATA_KEY

    drawn = graph.get_graph(xray=1)
    for key, node in list(drawn.nodes.items()):
        metadata = {
            name: value
            for name, value in (node.metadata or {}).items()
            if name not in (PHASE_METADATA_KEY, STEP_METADATA_KEY)
        }
        drawn.nodes[key] = node._replace(metadata=metadata or None)
    return str(drawn.draw_mermaid()).rstrip()


def render_diagrams(built: Sequence[tuple[GraphEntry, Any | None]]) -> str:
    """Render the generated part of ``docs/ARCHITECTURE.md``: one Mermaid diagram per graph."""
    lines = [BEGIN, ""]
    for entry, graph in built:
        lines += [f"### {entry.title}", ""]
        if graph is None:
            lines += ["Not available in this build.", ""]
            continue
        lines += ["```mermaid", _diagram(graph), "```", ""]
    lines.append(END)
    return "\n".join(lines)


def splice(document: str, generated: str) -> str:
    """Replace the generated block of ``document`` (or append it when absent)."""
    if BEGIN in document and END in document:
        head, rest = document.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        return f"{head}{generated}{tail}"
    return document.rstrip("\n") + "\n\n" + generated + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tool; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail when the files are out of date")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
    warnings: list[str] = []
    built = [(entry, _build(entry, warnings)) for entry in GRAPHS]
    for warning in warnings:
        print(f"list_steps: {warning}", file=sys.stderr)
    steps = render_steps(built)
    architecture_old = ARCHITECTURE_PATH.read_text() if ARCHITECTURE_PATH.exists() else ""
    architecture = splice(architecture_old, render_diagrams(built))
    if args.check:
        current = STEPS_PATH.read_text() if STEPS_PATH.exists() else ""
        stale = [
            str(path.relative_to(REPO_ROOT))
            for path, new, old in (
                (STEPS_PATH, steps, current),
                (ARCHITECTURE_PATH, architecture, architecture_old),
            )
            if new != old
        ]
        if stale or warnings:
            for path in stale:
                print(f"{path} is out of date; run tools/list_steps.py", file=sys.stderr)
            return 1
        print("steps and diagrams are up to date")
        return 0
    STEPS_PATH.write_text(steps)
    ARCHITECTURE_PATH.write_text(architecture)
    print(
        f"wrote {STEPS_PATH.relative_to(REPO_ROOT)} and {ARCHITECTURE_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
