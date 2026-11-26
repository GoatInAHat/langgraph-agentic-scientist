"""The investigation phase: ideation, attempts, workers and the no-winner downstream selection.

The graphs (each a factory returning a compiled graph, usable alone or mounted as a subgraph):

* :func:`build_investigation_graph`: the campaign over one or more workers, in rounds;
* :func:`build_campaign_worker_graph`: one worker's idea and attempt cycles;
* :func:`build_ideation_graph`: one cycle's idea;
* :func:`build_attempt_graph`: one idea's attempt, harness-evaluated or self-evaluated;
* :func:`build_downstream_selection_graph`: the no-winner fallback program.

:class:`HarnessRubricVerifier` is the default
:class:`~langgraph_agentic_scientist.core.interfaces.RubricVerifier`.
"""

from ..core.interfaces import RubricVerifier
from .attempt import build_attempt_graph
from .downstream import build_downstream_selection_graph
from .graph import build_investigation_graph
from .harness_eval import HarnessRubricVerifier
from .ideas import IdeaDraft
from .ideation import build_ideation_graph
from .models import (
    AttemptInput,
    AttemptOutput,
    AttemptResult,
    DownstreamInput,
    DownstreamOutput,
    IdeationInput,
    IdeationOutput,
    IdeationResult,
    InvestigationInput,
    InvestigationOutput,
    OuterState,
    WorkerInput,
    WorkerOutput,
    WorkerResult,
)
from .worker import build_campaign_worker_graph

__all__ = [
    "AttemptInput",
    "AttemptOutput",
    "AttemptResult",
    "DownstreamInput",
    "DownstreamOutput",
    "HarnessRubricVerifier",
    "IdeaDraft",
    "IdeationInput",
    "IdeationOutput",
    "IdeationResult",
    "InvestigationInput",
    "InvestigationOutput",
    "OuterState",
    "RubricVerifier",
    "WorkerInput",
    "WorkerOutput",
    "WorkerResult",
    "build_attempt_graph",
    "build_campaign_worker_graph",
    "build_downstream_selection_graph",
    "build_ideation_graph",
    "build_investigation_graph",
]
