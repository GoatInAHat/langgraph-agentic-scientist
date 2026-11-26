"""Concept figures (``lib/concept_figure.sh``) and the optional Autoillustrate renderer.

Upstream's orchestrator, not the writing agent, renders the paper's one concept figure: after
each writer round it resolves the request from ``paper_strategy.json`` (the one
``figure_specs`` entry whose ``preferred_generator`` is ``generate-concept-figure``), renders
it with the outside Autoillustrate CLI into ``figures/<figure_id>.png``, keeps a private copy,
and re-asserts that copy when a later repair overwrites the placeholder. Rendering is
non-gating: a failure is recorded and the manuscript compiles with the writer's placeholder.

The port keeps that transaction (:func:`prepare_concept_figure`) and moves the outside tool
behind :class:`~langgraph_agentic_scientist.core.interfaces.ConceptFigureRenderer`. The default
renderer is ``None``: no figure is rendered, and the writer prompts say so (adaptation A8).
:class:`AutoillustrateRenderer` runs the ``autoillustrate`` CLI through ``Workspace.run`` when
it is installed; Autoillustrate itself drives an agent CLI and an image model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from ..core.interfaces import ConceptFigureRequest, ConceptFigureResult, FigureAvailability
from ..core.state import Event
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .contexts import read_json_or_none
from .models import ConceptFigureState

if TYPE_CHECKING:
    from ..context import ScientistContext

CONCEPT_FIGURE_GENERATOR = "generate-concept-figure"
CONCEPT_FIGURE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
STASH_DIR = ".concept_figure"


def select_mode(preferred: str, budget_seconds: int | None, *, full: int, medium: int) -> str:
    """``select_concept_figure_mode``: step down from ``full`` when the budget is short."""
    if budget_seconds is None:
        return preferred
    if preferred == "full" and budget_seconds >= full:
        return "full"
    if preferred != "light" and budget_seconds >= medium:
        return "medium"
    return "light"


async def resolve_concept_figure_request(
    workspace: Workspace, paper_dir: str
) -> dict[str, str] | None:
    """``resolve_concept_figure_request``: the strategy's concept-figure request, or ``None``.

    Raises:
        ValueError: When the strategy asks for more than one concept figure or names an
            invalid figure id (upstream raises before anything is rendered).
    """
    strategy = await read_json_or_none(workspace, join(paper_dir, "paper_strategy.json"))
    if not isinstance(strategy, Mapping):
        return None
    specs = strategy.get("figure_specs")
    if not isinstance(specs, list):
        return None
    concept = [
        spec
        for spec in specs
        if isinstance(spec, Mapping)
        and str(spec.get("preferred_generator", "")).strip() == CONCEPT_FIGURE_GENERATOR
        and spec.get("required") is not False
    ]
    if len(concept) > 1:
        raise ValueError(
            f"paper_strategy.json may contain at most one {CONCEPT_FIGURE_GENERATOR} spec"
        )
    for spec in concept:
        figure_id = str(spec.get("figure_id", "")).strip()
        if not figure_id:
            continue
        if not CONCEPT_FIGURE_ID_RE.fullmatch(figure_id):
            raise ValueError(
                "paper_strategy.json contains invalid figure_id for "
                f"{CONCEPT_FIGURE_GENERATOR}: {figure_id!r}"
            )
        purpose = str(spec.get("purpose", "")).strip()
        span = str(spec.get("preferred_span", "")).strip().lower().replace("-", "_")
        return {
            "figure_id": figure_id,
            "output_rel": f"figures/{figure_id}.png",
            "instructions": purpose or f"Overview diagram for the figure '{figure_id}'.",
            "placement": "full-width" if span == "full_width" else "half-width",
        }
    return None


def request_fingerprint(request: Mapping[str, str]) -> str:
    """The request identity upstream hashes (``sort_keys``, compact separators)."""
    return hashlib.sha256(
        json.dumps(dict(request), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def preserve_fingerprint(workspace: Workspace, paper_dir: str) -> str | None:
    """A fingerprint of the concept state (the strategy and the figures directory)."""
    hasher = hashlib.sha256()
    strategy = join(paper_dir, "paper_strategy.json")
    if await workspace.exists(strategy):
        hasher.update(b"strategy\0" + (await workspace.read_bytes(strategy)))
    figures = join(paper_dir, "figures")
    if await workspace.exists(figures):
        for entry in await workspace.list_tree(figures):
            hasher.update(f"{entry.path}\0{entry.kind}\0{entry.sha256}\n".encode())
    return hasher.hexdigest()


def initial_state(
    ctx: ScientistContext,
    *,
    preserve: str | None,
    budget_seconds: int | None,
    full: int,
    medium: int,
    light: int,
) -> ConceptFigureState:
    """``initialize_concept_figure_step``: availability, mode and the preserve snapshot."""
    if ctx.figure_renderer is None:
        return ConceptFigureState(
            available=False,
            unavailable_reason="no concept-figure renderer is configured",
            preserve_fingerprint=preserve,
        )
    if budget_seconds is not None and budget_seconds < light:
        return ConceptFigureState(
            available=False,
            unavailable_reason="concept-figure workflow deadline budget exhausted",
            preserve_fingerprint=preserve,
        )
    mode = select_mode("full", budget_seconds, full=full, medium=medium)
    return ConceptFigureState(
        available=True,
        mode=mode,
        preserve_fingerprint=preserve,
    )


def stash_path(paper_dir: str, figure_id: str) -> str:
    """Where the rendered figure is kept so that it can be re-asserted (outside the paper)."""
    parent = paper_dir.rsplit("/", 1)[0] if "/" in paper_dir else ""
    name = paper_dir.rsplit("/", 1)[-1]
    return (
        join(parent, STASH_DIR, name, f"{figure_id}.png")
        if parent
        else join(STASH_DIR, name, f"{figure_id}.png")
    )


async def prepare_concept_figure(
    ctx: ScientistContext,
    *,
    paper_dir: str,
    concept: ConceptFigureState,
    refresh_mode: str,
) -> tuple[ConceptFigureState, list[Event]]:
    """``prepare_concept_figure_for_validation``: render, re-assert or record the concept figure.

    Never raises for a renderer failure: the step is non-gating, as upstream's.
    """
    ws = ctx.workspace
    events: list[Event] = []
    state = concept.model_copy()
    if refresh_mode == "preserve":
        current = await preserve_fingerprint(ws, paper_dir)
        if state.preserve_fingerprint is not None and current == state.preserve_fingerprint:
            events.append(Event(kind="concept_figure_preserved", phase="paper", step="figures"))
            return state, events
    try:
        request = await resolve_concept_figure_request(ws, paper_dir)
    except ValueError as exc:
        events.append(
            Event(
                kind="concept_figure_skipped",
                phase="paper",
                step="figures",
                data={"reason": f"concept-figure resolver failed: {exc}"},
            )
        )
        return ConceptFigureState(
            available=state.available,
            unavailable_reason=state.unavailable_reason,
            mode=state.mode,
            preserve_fingerprint=state.preserve_fingerprint,
        ), events
    if request is None:
        return ConceptFigureState(
            available=state.available,
            unavailable_reason=state.unavailable_reason,
            mode=state.mode,
            preserve_fingerprint=state.preserve_fingerprint,
        ), events
    fingerprint = request_fingerprint(request)
    if not state.available or ctx.figure_renderer is None:
        if not (state.attempted and state.request_fingerprint == fingerprint):
            state.attempted = True
            state.request_fingerprint = fingerprint
            state.failure_active = True
            state.failure_reason = (
                state.unavailable_reason or "concept-figure private state unavailable"
            )
            events.append(
                Event(
                    kind="concept_figure_unavailable",
                    phase="paper",
                    step="figures",
                    data={"figure_id": request["figure_id"], "reason": state.failure_reason},
                )
            )
        return state, events
    target = join(paper_dir, request["output_rel"])
    if state.attempted and state.request_fingerprint == fingerprint:
        if state.failure_active:
            return state, events
        await reassert_concept_figure(ws, state, target, events)
        return state, events
    result = await ctx.figure_renderer.render(
        ConceptFigureRequest(
            paper_dir=paper_dir,
            instructions=request["instructions"],
            mode=state.mode,
            output_path=request["output_rel"],
            metadata={"figure_id": request["figure_id"], "placement": request["placement"]},
        ),
        workspace=ws,
    )
    state.attempted = True
    state.request_fingerprint = fingerprint
    state.repair_authorized = False
    if not result.success or not await ws.exists(target):
        state.failure_active = True
        state.failure_reason = result.detail or "concept-figure renderer failed"
        events.append(
            Event(
                kind="concept_figure_failed",
                phase="paper",
                step="figures",
                data={"figure_id": request["figure_id"], "reason": state.failure_reason},
            )
        )
        return state, events
    data = await ws.read_bytes(target)
    stash = stash_path(paper_dir, request["figure_id"])
    await ws.write_bytes(stash, data)
    state.failure_active = False
    state.failure_reason = ""
    state.rendered_path = stash
    state.rendered_sha256 = hashlib.sha256(data).hexdigest()
    events.append(
        Event(
            kind="concept_figure_rendered",
            phase="paper",
            step="figures",
            data={"figure_id": request["figure_id"], "path": target},
        )
    )
    if ctx.paper_toolchain is not None and await ws.exists(join(paper_dir, "paper.tex")):
        try:
            await ctx.paper_toolchain.compile(paper_dir, workspace=ws)
        except Exception as exc:  # non-gating, as upstream's best-effort compile
            events.append(
                Event(
                    kind="concept_figure_compile_failed",
                    phase="paper",
                    step="figures",
                    data={"detail": str(exc)[:2000]},
                )
            )
    return state, events


async def reassert_concept_figure(
    workspace: Workspace, state: ConceptFigureState, target: str, events: list[Event]
) -> None:
    """``reassert_concept_figure``: restore the rendered figure a repair overwrote, unless an
    evidence-audit finding authorised repairing it."""
    if not state.rendered_path or not await workspace.exists(state.rendered_path):
        return
    if state.repair_authorized:
        events.append(
            Event(kind="concept_figure_fallback_preserved", phase="paper", step="figures")
        )
        return
    stashed = await workspace.read_bytes(state.rendered_path)
    if await workspace.exists(target) and await workspace.read_bytes(target) == stashed:
        return
    await workspace.write_bytes(target, stashed)
    events.append(
        Event(
            kind="concept_figure_reasserted", phase="paper", step="figures", data={"path": target}
        )
    )


async def authorize_concept_figure_repair(
    workspace: Workspace, paper_dir: str, state: ConceptFigureState
) -> ConceptFigureState:
    """``authorize_concept_figure_repair``: let the next repair replace the rendered figure when
    the evidence audit reported findings about it."""
    if not state.attempted or state.failure_active or not state.rendered_path:
        return state
    try:
        request = await resolve_concept_figure_request(workspace, paper_dir)
    except ValueError:
        return state
    if request is None or request_fingerprint(request) != state.request_fingerprint:
        return state
    feedback = await read_json_or_none(
        workspace, join(paper_dir, "paper_evidence_audit_feedback.json")
    )
    findings = feedback.get("findings") if isinstance(feedback, Mapping) else None
    if not isinstance(findings, list):
        return state
    figure_id = request["figure_id"]
    if any(figure_id in json.dumps(item, sort_keys=True) for item in findings):
        updated = state.model_copy()
        updated.repair_authorized = True
        return updated
    return state


class AutoillustrateRenderer:
    """A :class:`~langgraph_agentic_scientist.core.interfaces.ConceptFigureRenderer` over the
    outside ``autoillustrate`` CLI (upstream ``run_concept_figure_step``).

    The renderer runs ``autoillustrate ./paper.tex --output ./figures/<id>.png --placement ...
    --agent <agent> --mode <mode> --prompt <text>`` in the paper directory through
    ``Workspace.run``. Autoillustrate drives an agent CLI and, for its restyle pass, an image model
    (OpenAI by default, Gemini as fallback); without either key it runs with ``--skip-enhance``.
    The ``--prompt`` text is upstream's (``prompts/workflow/paper/concept_figure``) followed by
    the strategy's figure purpose.

    Args:
        agent: The agent CLI Autoillustrate dispatches (upstream ``--agent``).
        executable: The ``autoillustrate`` command.
        timeout_seconds: Wall clock of one render (upstream default 9000 s).
        environment: Environment of the command; defaults to the host environment.
        prompts: The registry holding the availability texts and the prompt prefix.
    """

    def __init__(
        self,
        *,
        agent: str = "claude",
        executable: str = "autoillustrate",
        timeout_seconds: float = 9000,
        environment: Mapping[str, str] | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        self.agent = agent
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.environment = dict(environment) if environment is not None else dict(os.environ)
        self.prompts = prompts or PromptRegistry.default()

    def _has(self, name: str) -> bool:
        return bool(self.environment.get(name, "").strip())

    def availability(self) -> FigureAvailability:
        """Upstream's key contexts, chosen by the keys present in the renderer's environment."""
        openai = self._has("OPENAI_API_KEY")
        gemini = self._has("GEMINI_API_KEY")
        text = self.prompts.text
        return FigureAvailability(
            openai=openai,
            gemini=gemini,
            openai_context=text(
                "workflow/paper/openai_key_available"
                if openai
                else "workflow/paper/openai_key_missing"
            ),
            gemini_context=text(
                "workflow/paper/gemini_key_available"
                if gemini
                else "workflow/paper/gemini_key_missing"
            ),
        )

    async def render(
        self, request: ConceptFigureRequest, *, workspace: Workspace
    ) -> ConceptFigureResult:
        """Render one concept figure; a missing CLI or a failed run is a non-gating failure."""
        availability = self.availability()
        prefix = self.prompts.text("workflow/paper/concept_figure").strip()
        argv = [
            self.executable,
            "./paper.tex",
            "--output",
            f"./{request.output_path}",
            "--placement",
            str(request.metadata.get("placement", "half-width")),
            "--agent",
            self.agent,
            "--mode",
            request.mode,
            "--prompt",
            f"{prefix} {request.instructions}".strip(),
        ]
        if not availability.openai and not availability.gemini:
            argv.append("--skip-enhance")
        try:
            result = await workspace.run(
                argv,
                cwd=request.paper_dir,
                env=self.environment,
                timeout=self.timeout_seconds,
            )
        except (OSError, FileNotFoundError) as exc:
            return ConceptFigureResult(success=False, detail=f"autoillustrate unavailable: {exc}")
        target = join(request.paper_dir, request.output_path)
        if result.returncode != 0 or not await workspace.exists(target):
            detail = (result.stderr or result.stdout or "").strip()[-2000:]
            return ConceptFigureResult(
                success=False, detail=f"autoillustrate exited {result.returncode}: {detail}"
            )
        return ConceptFigureResult(success=True, path=target)


RefreshMode = Literal["auto", "preserve", "refresh"]


def figure_step_updates(state: ConceptFigureState, events: list[Event]) -> dict[str, Any]:
    """The state update of the figures step."""
    return {"concept": state, "events": events}


__all__ = [
    "CONCEPT_FIGURE_GENERATOR",
    "AutoillustrateRenderer",
    "RefreshMode",
    "authorize_concept_figure_repair",
    "figure_step_updates",
    "initial_state",
    "prepare_concept_figure",
    "preserve_fingerprint",
    "reassert_concept_figure",
    "request_fingerprint",
    "resolve_concept_figure_request",
    "select_mode",
    "stash_path",
]
