"""Exceptions raised by the package.

Every exception derives from :class:`ScientistError`, so a caller can catch the package's own
failures without catching unrelated ones.
"""

from __future__ import annotations

from collections.abc import Sequence


class ScientistError(Exception):
    """Base class of every exception the package raises on purpose."""


class ConfigurationMismatch(ScientistError):
    """A resumed run was given a method configuration or prompts that differ from the recorded ones.

    Attributes:
        fields: The dotted names of the configuration fields, or the prompt keys, that differ.
    """

    def __init__(self, fields: Sequence[str], *, detail: str = "") -> None:
        self.fields = tuple(fields)
        names = ", ".join(self.fields) or "(unknown)"
        message = f"the run configuration differs from the one recorded in the checkpoint: {names}"
        if detail:
            message = f"{message}; {detail}"
        super().__init__(message)


class IncompatibleCheckpoint(ScientistError):
    """A checkpoint was written by a release whose state schema this release cannot read."""

    def __init__(self, recorded: int, current: int) -> None:
        self.recorded = recorded
        self.current = current
        super().__init__(
            f"checkpoint state schema_version {recorded} is not supported by this release "
            f"(schema_version {current}); no migration covers it"
        )


class WorkspacePathError(ScientistError, ValueError):
    """A workspace path is absolute, escapes the workspace, or is otherwise unsafe."""


class PromptNotFound(ScientistError, KeyError):
    """A prompt key is not in the registry."""


class ProgramSpecError(ScientistError, ValueError):
    """A ``program_spec.toml`` is missing or invalid (upstream ``ProgramSpecError``)."""


class TaskContractError(ScientistError, ValueError):
    """A ``task_contract.json`` or one of its declared artifacts is invalid (upstream rules)."""


class EvaluatorOutputContractError(ScientistError, ValueError):
    """An evaluator produced artifacts incompatible with the host contract.

    This is upstream's terminal ``evaluator_output_contract_invalid`` category (exit 86).
    """

    category = "evaluator_output_contract_invalid"


class FinalEvaluationError(ScientistError, RuntimeError):
    """A sealed final evaluation could not run or could not be trusted (upstream exit 1)."""


class FinalEvaluationInfrastructureError(FinalEvaluationError):
    """Host infrastructure prevented the sealed final evaluation (upstream exit 4)."""


class FinalEvaluationAlreadyStarted(FinalEvaluationError):
    """A sealed final evaluation with this id started earlier and its result cannot be obtained.

    Upstream refuses to run the sealed evaluation a second time once ``invocation.json`` exists
    without a manifest; the port keeps that once-only rule.
    """

    def __init__(self, evaluation_id: str) -> None:
        self.evaluation_id = evaluation_id
        super().__init__(
            f"final evaluation {evaluation_id!r} already started; refusing to rerun the sealed "
            "evaluation"
        )


class AgentRunError(ScientistError, RuntimeError):
    """An agent session could not be run or recorded."""


class SkillNotFound(ScientistError, KeyError):
    """A skill name is not shipped with the package."""
