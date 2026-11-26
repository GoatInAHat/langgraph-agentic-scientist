"""The paper validation gate: validators, the LaTeX compile and the default toolchain.

:class:`WorkspacePaperToolchain` implements
:class:`~langgraph_agentic_scientist.core.interfaces.PaperToolchain` (upstream
``validate_and_compile_paper``). The individual validators are importable on their own; each takes
a :class:`~langgraph_agentic_scientist.paper.validation.files.PaperTree` (or, where a skill's
script needs it, a host directory path).
"""

from .figures import (
    classify_figure_validation_errors,
    has_actionable_figure_quality_warnings,
    validate_figure_manifest,
)
from .files import DirectoryTree, PaperTree, SnapshotTree
from .fingerprint import (
    validation_failure_fingerprint,
    validation_hard_error_fingerprint,
    validation_input_sha256,
)
from .latex import PaperValidationInfrastructureError
from .scope_audit import validate_method_scope_audit, validate_method_scope_audit_in
from .validator import WorkspacePaperToolchain, load_validation_tree

__all__ = [
    "DirectoryTree",
    "PaperTree",
    "PaperValidationInfrastructureError",
    "SnapshotTree",
    "WorkspacePaperToolchain",
    "classify_figure_validation_errors",
    "has_actionable_figure_quality_warnings",
    "load_validation_tree",
    "validate_figure_manifest",
    "validate_method_scope_audit",
    "validate_method_scope_audit_in",
    "validation_failure_fingerprint",
    "validation_hard_error_fingerprint",
    "validation_input_sha256",
]
