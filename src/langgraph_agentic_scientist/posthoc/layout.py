"""Where the post-hoc phase keeps its files (upstream's output layout, as workspace paths).

::

    <output>/
      _extracted_bundle/          stage 0
      _analysis_ready_bundle/     stage 1
      bootstrap_report.json
      generation/                 stages 2 and 3 (posthoc_bundle.json, posthoc_selection.json, ...)
      analyses/<analysis_id>/     stages 4 and 5 (task_bundle/, attempt_<n>/, analysis_result.json)
      ranking.json, latest_manifest.json, pipeline_summary.json     stage 6
"""

from __future__ import annotations

from dataclasses import dataclass

from ..workspace.paths import join, safe_relpath

DEFAULT_OUTPUT_DIR = "posthoc"


@dataclass(frozen=True)
class Layout:
    """The paths of one post-hoc run below its output directory."""

    output: str

    @classmethod
    def of(cls, output_dir: str | None) -> Layout:
        """Return the layout of ``output_dir`` (default ``posthoc``), refusing unsafe paths."""
        return cls(safe_relpath(output_dir or DEFAULT_OUTPUT_DIR, allow_root=False))

    @property
    def bundle(self) -> str:
        """The extracted bundle (stage 0)."""
        return join(self.output, "_extracted_bundle")

    @property
    def ready(self) -> str:
        """The analysis-ready bundle (stage 1)."""
        return join(self.output, "_analysis_ready_bundle")

    @property
    def generation(self) -> str:
        """The directory of the generator and the selector."""
        return join(self.output, "generation")

    @property
    def analyses(self) -> str:
        """The directory of the analysis workspaces."""
        return join(self.output, "analyses")

    @property
    def bootstrap_report(self) -> str:
        """The authoritative checkpoint and guardrail record."""
        return join(self.output, "bootstrap_report.json")

    def analysis(self, analysis_id: str) -> str:
        """The workspace of one analysis."""
        return join(self.analyses, analysis_id)

    def attempt(self, analysis_id: str, number: int) -> str:
        """The directory of one attempt of an analysis."""
        return join(self.analysis(analysis_id), f"attempt_{number}")


__all__ = ["DEFAULT_OUTPUT_DIR", "Layout"]
