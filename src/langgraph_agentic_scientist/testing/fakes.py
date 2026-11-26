"""Fakes of every outside dependency, for tests that run offline.

* :class:`FakeEvaluator` and :class:`FakeFinalEvaluator` honour the evaluator contract
  (idempotence by id, re-attachment, once-only) and record their calls.
* :class:`FakeTaskGate`, :class:`FakeRetriever`, :class:`FakePaperToolchain` and
  :class:`FakeFigureRenderer` return scripted results.
* :class:`MemoryWorkspace` is an in-memory
  :class:`~langgraph_agentic_scientist.workspace.Workspace` whose agent backend is an in-memory
  sandbox with a scripted ``execute``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)

from ..core.errors import FinalEvaluationAlreadyStarted
from ..core.interfaces import (
    CompileResult,
    ConceptFigureRequest,
    ConceptFigureResult,
    FigureAvailability,
    GateCheck,
    GateReport,
    ReferenceQuery,
    ReferenceRecord,
)
from ..core.results import ValidationReport
from ..core.state import TaskRef
from ..evaluation.protocol import (
    ComputeUsage,
    EvaluationRequest,
    EvaluationResult,
    FinalEvaluationRequest,
    FinalEvaluationResult,
)
from ..workspace.digest import sha256_bytes, tree_digest
from ..workspace.paths import safe_relpath, task_copy_path_is_ignored
from ..workspace.protocol import CommandResult, FileEntry, SnapshotRef, Workspace
from ..workspace.readonly import ReadOnlyBackend

EvaluationAnswer = EvaluationResult | Callable[[EvaluationRequest], EvaluationResult]


def completed_result(
    request: EvaluationRequest, *, combined_score: float = 1.0, correct: bool = True
) -> EvaluationResult:
    """Return a completed evaluation result for a request."""
    return EvaluationResult(
        evaluation_id=request.evaluation_id,
        status="completed",
        metrics={"combined_score": combined_score},
        correct=correct,
        summary=f"returncode=0; combined_score={combined_score}; correct={str(correct).lower()}",
        output_dir=f".evaluations/{request.evaluation_id}/results",
        returncode=0,
    )


class FakeEvaluator:
    """A scripted :class:`~langgraph_agentic_scientist.evaluation.Evaluator`.

    Args:
        results: Answers keyed by evaluation id, program path or program digest (looked up in
            that order); a value is an ``EvaluationResult`` or a callable of the request.
        default: Answer for requests the table does not name (default: completed, score 1.0).
        fail_once: Evaluation ids whose first call starts the evaluation and then raises
            ``ConnectionError``, as if the caller crashed while the evaluation ran.
        reattach: When true, a later call for such an id re-attaches to the started evaluation
            and returns its result without evaluating again; when false it returns
            ``status="interrupted"``.
        isolated: What the evaluator reports as ``isolated``.

    Attributes:
        calls: Every request received, in order.
        runs: How many times each id was actually evaluated (``1`` at most, by contract).
    """

    def __init__(
        self,
        results: Mapping[str, EvaluationAnswer] | None = None,
        *,
        default: EvaluationAnswer | None = None,
        fail_once: Sequence[str] = (),
        reattach: bool = False,
        isolated: bool = True,
    ) -> None:
        self.results = dict(results or {})
        self.default = default
        self.fail_once = set(fail_once)
        self.reattach = reattach
        self._isolated = isolated
        self.calls: list[EvaluationRequest] = []
        self.runs: dict[str, int] = defaultdict(int)
        self._recorded: dict[str, EvaluationResult] = {}
        self._started: dict[str, EvaluationResult] = {}

    @property
    def isolated(self) -> bool:
        """What the fake was told to report."""
        return self._isolated

    def _answer(self, request: EvaluationRequest) -> EvaluationResult:
        answer = (
            self.results.get(request.evaluation_id)
            or self.results.get(request.program.path)
            or (self.results.get(request.program.digest) if request.program.digest else None)
            or self.default
        )
        if answer is None:
            return completed_result(request)
        result = answer(request) if callable(answer) else answer
        return result.model_copy(update={"evaluation_id": request.evaluation_id})

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate once per id; re-attach or report ``interrupted`` for a started id."""
        self.calls.append(request)
        key = request.evaluation_id
        if key in self._recorded:
            return self._recorded[key]
        if key in self._started:
            if self.reattach:
                result = self._started.pop(key)
            else:
                self._started.pop(key)
                result = EvaluationResult(
                    evaluation_id=key,
                    status="interrupted",
                    summary="the evaluation started earlier and its result is lost",
                )
            self._recorded[key] = result
            return result
        self.runs[key] += 1
        result = self._answer(request)
        if key in self.fail_once:
            self.fail_once.discard(key)
            self._started[key] = result
            raise ConnectionError(f"caller lost while evaluation {key} was running")
        self._recorded[key] = result
        return result


class FakeFinalEvaluator:
    """A scripted :class:`~langgraph_agentic_scientist.evaluation.FinalEvaluator`, once per id.

    Args:
        result: The result of every sealed evaluation (a callable of the request, or a value).
        fail_once: When true the first call starts the evaluation and raises ``ConnectionError``;
            a later call for that id raises
            :class:`~langgraph_agentic_scientist.core.errors.FinalEvaluationAlreadyStarted`.

    Attributes:
        calls: Every request received (with its sealed inputs).
        runs: How many sealed evaluations ran.
    """

    def __init__(
        self,
        result: FinalEvaluationResult
        | Callable[[FinalEvaluationRequest], FinalEvaluationResult]
        | None = None,
        *,
        fail_once: bool = False,
        isolated: bool = True,
    ) -> None:
        self.result = result
        self.fail_once = fail_once
        self._isolated = isolated
        self.calls: list[FinalEvaluationRequest] = []
        self.runs = 0
        self._recorded: dict[str, FinalEvaluationResult] = {}
        self._started: set[str] = set()

    @property
    def isolated(self) -> bool:
        """What the fake was told to report."""
        return self._isolated

    def _answer(self, request: FinalEvaluationRequest) -> FinalEvaluationResult:
        if callable(self.result):
            return self.result(request)
        if self.result is not None:
            return self.result
        frozen = request.frozen
        return FinalEvaluationResult(
            correct=True,
            outcome="qualified",
            metrics={"combined_score": 1.0},
            declared_final_metrics=["combined_score"],
            task_tree_sha256=frozen.task_tree_sha256,
            evaluator_sha256=frozen.evaluator_sha256,
            program_tree_sha256=frozen.program_tree_sha256,
            results_tree_sha256=sha256_bytes(b"fake results"),
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            compute_usage=ComputeUsage(
                wall_clock_limit_seconds=7200,
                elapsed_seconds=1.0,
                max_rpc_calls=100_000,
                rpc_calls=None,
                outcome="completed",
            ),
        )

    async def evaluate_sealed(self, request: FinalEvaluationRequest) -> FinalEvaluationResult:
        """Run once per id; replay the recorded result; refuse a started id."""
        self.calls.append(request)
        key = request.evaluation_id
        if key in self._recorded:
            return self._recorded[key]
        if key in self._started:
            raise FinalEvaluationAlreadyStarted(key)
        self.runs += 1
        if self.fail_once:
            self.fail_once = False
            self._started.add(key)
            raise ConnectionError(f"caller lost while final evaluation {key} was running")
        result = self._answer(request)
        self._recorded[key] = result
        return result


class FakeTaskGate:
    """A scripted :class:`~langgraph_agentic_scientist.core.interfaces.TaskGate`.

    Args:
        reports: Per check, the reports to return in order (the last one repeats), or a callable
            ``(check, call_number) -> GateReport``.
    """

    def __init__(
        self,
        reports: Mapping[str, Sequence[GateReport]]
        | Callable[[str, int], GateReport]
        | None = None,
    ) -> None:
        self.reports = reports or {}
        self.calls: list[tuple[str, str]] = []
        self._counts: dict[str, int] = defaultdict(int)

    async def run_check(
        self, check: GateCheck, *, task: TaskRef, workspace: Workspace, evaluation_id: str
    ) -> GateReport:
        """Return the next scripted report of ``check`` (passed when none is scripted)."""
        self.calls.append((check, evaluation_id))
        number = self._counts[check]
        self._counts[check] += 1
        if callable(self.reports):
            return self.reports(check, number)
        scripted = list(self.reports.get(check, ()))
        if not scripted:
            return GateReport(check=check, passed=True)
        return scripted[min(number, len(scripted) - 1)]


class FakeRetriever:
    """A scripted :class:`~langgraph_agentic_scientist.core.interfaces.ReferenceRetriever`."""

    def __init__(
        self,
        records: Mapping[str, Sequence[ReferenceRecord]] | None = None,
        *,
        default: Sequence[ReferenceRecord] = (),
    ) -> None:
        self.records = dict(records or {})
        self.default = list(default)
        self.queries: list[ReferenceQuery] = []

    async def search(self, query: ReferenceQuery, *, workspace: Workspace) -> list[ReferenceRecord]:
        """Return the records scripted for the query text (or the default records)."""
        self.queries.append(query)
        return list(self.records.get(query.query, self.default))[: query.limit]


def text_pdf(text: str, *, width: int = 612, height: int = 792) -> bytes:
    """Return a minimal one-page PDF that shows ``text`` in Helvetica (ASCII; wrapped lines).

    Written by hand so that tests need no PDF library to make one; any PDF reader extracts the
    text.
    """
    words = text.encode("ascii", "replace").decode("ascii").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > 90:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)

    def escape(line: str) -> str:
        return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    body = ["BT", "/F1 10 Tf", "12 TL", f"40 {height - 50} Td"]
    for line in lines[:60]:
        body.append(f"({escape(line)}) Tj T*")
    body.append("ET")
    stream = "\n".join(body).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


class FakePaperToolchain:
    """A :class:`~langgraph_agentic_scientist.core.interfaces.PaperToolchain` without LaTeX.

    ``compile`` writes a one-page PDF whose text is ``pdf_text`` (so a PDF reader, such as the
    review phase's, finds it); ``validate`` returns the scripted reports in order (the last one
    repeats; passed when none is scripted).
    """

    def __init__(
        self,
        reports: Sequence[ValidationReport] | Callable[[str, int], ValidationReport] | None = None,
        *,
        pdf_text: str = "Stub paper text. " * 20,
    ) -> None:
        self.reports = reports
        self.text = pdf_text
        self.validations: list[str] = []
        self.compilations: list[str] = []

    async def validate(
        self, paper_dir: str, *, workspace: Workspace, options: dict[str, Any]
    ) -> ValidationReport:
        """Return the next scripted validation report and write the stub PDF."""
        number = len(self.validations)
        self.validations.append(paper_dir)
        await self.compile(paper_dir, workspace=workspace)
        if callable(self.reports):
            return self.reports(paper_dir, number)
        if not self.reports:
            return ValidationReport(passed=True, pdf_path=f"{paper_dir}/paper.pdf")
        return list(self.reports)[min(number, len(self.reports) - 1)]

    async def compile(self, paper_dir: str, *, workspace: Workspace) -> CompileResult:
        """Write ``<paper_dir>/paper.pdf`` as a stub."""
        self.compilations.append(paper_dir)
        await workspace.write_bytes(f"{paper_dir}/paper.pdf", text_pdf(self.text))
        return CompileResult(success=True, pdf_path=f"{paper_dir}/paper.pdf", page_count=1)

    async def pdf_text(self, pdf_path: str, *, workspace: Workspace) -> str:
        """Return the configured text."""
        return self.text


class FakeFigureRenderer:
    """A :class:`~langgraph_agentic_scientist.core.interfaces.ConceptFigureRenderer` stub."""

    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.requests: list[ConceptFigureRequest] = []

    def availability(self) -> FigureAvailability:
        """Report both backends as available."""
        return FigureAvailability(
            openai=True,
            gemini=True,
            openai_context="A concept-figure renderer (fake) is configured.",
            gemini_context="A concept-figure renderer (fake) is configured.",
        )

    async def render(
        self, request: ConceptFigureRequest, *, workspace: Workspace
    ) -> ConceptFigureResult:
        """Write a stub PNG (or fail, when configured to)."""
        self.requests.append(request)
        if not self.success:
            return ConceptFigureResult(success=False, detail="fake renderer configured to fail")
        path = f"{request.paper_dir}/{request.output_path}"
        await workspace.write_bytes(path, self.PNG)
        return ConceptFigureResult(success=True, path=path)


# --- in-memory workspace ----------------------------------------------------------------------

ExecuteHandler = Callable[[str, int | None], ExecuteResponse]
CommandHandler = Callable[
    [Sequence[str], str, Mapping[str, str] | None], CommandResult | Awaitable[CommandResult]
]


def _unavailable(command: str, timeout: int | None) -> ExecuteResponse:
    return ExecuteResponse(
        output=f"{command.split()[0] if command.split() else command}: command not available",
        exit_code=127,
    )


class MemoryBackend(SandboxBackendProtocol):
    """An in-memory Deep Agents sandbox over a :class:`MemoryWorkspace` directory."""

    def __init__(
        self,
        workspace: MemoryWorkspace,
        root: str,
        *,
        execute: ExecuteHandler | None = None,
        execute_timeout: int = 600,
        execute_max_timeout: int = 7200,
    ) -> None:
        self.workspace = workspace
        self.root = safe_relpath(root)
        self.handler = execute or _unavailable
        self.execute_timeout = execute_timeout
        self.execute_max_timeout = execute_max_timeout
        self.commands: list[tuple[str, int]] = []
        self._id = f"memory-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """The sandbox id."""
        return self._id

    def _key(self, path: str) -> str:
        relative = safe_relpath(path)
        return f"{self.root}/{relative}".strip("/") if self.root else relative

    def _virtual(self, key: str) -> str:
        relative = key[len(self.root) :].lstrip("/") if self.root else key
        return "/" + relative

    def _files_under(self, key: str) -> list[str]:
        prefix = f"{key}/" if key else ""
        return sorted(path for path in self.workspace.files if path.startswith(prefix))

    def ls(self, path: str) -> LsResult:
        """List one directory level."""
        key = self._key(path)
        entries: dict[str, FileInfo] = {}
        for file_key in self._files_under(key):
            rest = file_key[len(key) :].lstrip("/") if key else file_key
            head, _, tail = rest.partition("/")
            child = f"{key}/{head}" if key else head
            if tail:
                entries[child] = FileInfo(path=self._virtual(child), is_dir=True)
            else:
                entries[child] = FileInfo(
                    path=self._virtual(child),
                    is_dir=False,
                    size=len(self.workspace.files[file_key]),
                )
        return LsResult(entries=[entries[name] for name in sorted(entries)])

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a window of lines of a file."""
        key = self._key(file_path)
        if key not in self.workspace.files:
            return ReadResult(error=f"Error: File '{file_path}' not found")
        text = self.workspace.files[key].decode("utf-8", errors="replace")
        lines = text.splitlines()
        offset = max(offset, 0)
        if limit <= 0:
            return ReadResult(
                file_data=FileData(content="", encoding="utf-8"), no_lines_requested=True
            )
        window = lines[offset : offset + limit]
        if not window:
            return ReadResult(file_data=FileData(content="", encoding="utf-8"))
        end = offset + len(window)
        return ReadResult(
            file_data=FileData(content="\n".join(window), encoding="utf-8"),
            start_line=offset + 1,
            end_line=end,
            total_lines=len(lines),
            next_offset=end if end < len(lines) else None,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file."""
        key = self._key(file_path)
        self.workspace.files[key] = content.encode("utf-8")
        return WriteResult(path=self._virtual(key))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text in a file."""
        key = self._key(file_path)
        if key not in self.workspace.files:
            return EditResult(error=f"Error: File '{file_path}' not found")
        text = self.workspace.files[key].decode("utf-8")
        count = text.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if count > 1 and not replace_all:
            return EditResult(error=f"Error: String appears {count} times; use replace_all")
        text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        self.workspace.files[key] = text.encode("utf-8")
        return EditResult(path=self._virtual(key), occurrences=count if replace_all else 1)

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file or a directory tree."""
        key = self._key(file_path)
        targets = [key] if key in self.workspace.files else self._files_under(key)
        if not targets:
            return DeleteResult(error=f"Error: '{file_path}' not found")
        for target in targets:
            self.workspace.files.pop(target, None)
        return DeleteResult(path=self._virtual(key))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files by glob pattern."""
        base = self._key(path or "/")
        matches = []
        for file_key in self._files_under(base):
            relative = file_key[len(base) :].lstrip("/") if base else file_key
            name = PurePosixPath(relative).name
            target = relative if "/" in pattern.lstrip("/") else name
            if fnmatch.fnmatch(target, pattern.lstrip("/")):
                matches.append(FileInfo(path=self._virtual(file_key), is_dir=False))
        return GlobResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal string."""
        base = self._key(path or "/")
        matches: list[GrepMatch] = []
        for file_key in self._files_under(base):
            if glob and not fnmatch.fnmatch(PurePosixPath(file_key).name, glob):
                continue
            text = self.workspace.files[file_key].decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=self._virtual(file_key), line=number, text=line))
        if max_count is not None and len(matches) > max_count:
            return GrepResult(matches=matches[:max_count], truncated=True)
        return GrepResult(matches=matches)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write files."""
        for path, data in files:
            self.workspace.files[self._key(path)] = data
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read files."""
        responses = []
        for path in paths:
            key = self._key(path)
            if key in self.workspace.files:
                responses.append(FileDownloadResponse(path=path, content=self.workspace.files[key]))
            else:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
        return responses

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run the scripted command handler (refusing timeouts above the maximum)."""
        if timeout is not None and timeout > self.execute_max_timeout:
            return ExecuteResponse(
                output=(
                    f"Error: timeout {timeout}s exceeds maximum allowed "
                    f"({self.execute_max_timeout}s)."
                ),
                exit_code=1,
            )
        effective = timeout if timeout is not None else self.execute_timeout
        self.commands.append((command, effective))
        return self.handler(command, effective)


class MemoryWorkspace:
    """An in-memory :class:`~langgraph_agentic_scientist.workspace.Workspace` for tests.

    Args:
        files: Initial files, keyed by workspace path.
        execute: Handler of agent ``execute`` calls ``(command, timeout) -> ExecuteResponse``.
        run: Handler of :meth:`run` calls ``(argv, cwd, env) -> CommandResult``.
        isolated: What the workspace reports as ``isolated``.

    Attributes:
        files: The files, keyed by workspace path (directories are implied).
        backends: Every agent backend handed out, in order.
    """

    def __init__(
        self,
        files: Mapping[str, bytes | str] | None = None,
        *,
        execute: ExecuteHandler | None = None,
        run: CommandHandler | None = None,
        isolated: bool = False,
    ) -> None:
        self.files: dict[str, bytes] = {
            safe_relpath(path): data.encode("utf-8") if isinstance(data, str) else data
            for path, data in (files or {}).items()
        }
        self.execute_handler = execute
        self.run_handler = run
        self._isolated = isolated
        self._snapshots: dict[str, dict[str, bytes]] = {}
        self.backends: list[BackendProtocol] = []

    @property
    def isolated(self) -> bool:
        """What the workspace was told to report."""
        return self._isolated

    def agent_backend(
        self,
        root: str,
        *,
        read_only: bool = False,
        execute_timeout: int = 600,
        execute_max_timeout: int = 7200,
        env: Mapping[str, str] | None = None,
    ) -> BackendProtocol:
        """Return an in-memory sandbox rooted at ``root`` (read-only when asked)."""
        inner = MemoryBackend(
            self,
            root,
            execute=self.execute_handler,
            execute_timeout=execute_timeout,
            execute_max_timeout=execute_max_timeout,
        )
        backend: BackendProtocol = ReadOnlyBackend(inner) if read_only else inner
        self.backends.append(backend)
        return backend

    def _children(self, path: str) -> list[str]:
        prefix = f"{path}/" if path else ""
        return sorted(key for key in self.files if key.startswith(prefix))

    async def read_bytes(self, path: str) -> bytes:
        """Return a file's bytes."""
        key = safe_relpath(path)
        if key not in self.files:
            raise FileNotFoundError(path)
        return self.files[key]

    async def write_bytes(self, path: str, data: bytes, *, exclusive: bool = False) -> None:
        """Write a file."""
        key = safe_relpath(path, allow_root=False)
        if exclusive and key in self.files:
            raise FileExistsError(path)
        self.files[key] = bytes(data)

    async def exists(self, path: str) -> bool:
        """Return whether a file or a directory exists."""
        return await self.stat(path) is not None

    async def stat(self, path: str) -> FileEntry | None:
        """Return a file or (implied) directory entry."""
        key = safe_relpath(path)
        if key in self.files:
            return FileEntry(path=key, size=len(self.files[key]), kind="file")
        if not key or self._children(key):
            return FileEntry(path=key, kind="dir")
        return None

    async def list_tree(self, path: str) -> list[FileEntry]:
        """List files and implied directories below ``path``."""
        base = safe_relpath(path)
        if base and not self._children(base):
            raise FileNotFoundError(path)
        entries: dict[str, FileEntry] = {}
        for key in self._children(base):
            relative = key[len(base) :].lstrip("/") if base else key
            parts = relative.split("/")
            for depth in range(1, len(parts)):
                directory = "/".join(parts[:depth])
                entries[directory] = FileEntry(path=directory, kind="dir")
            data = self.files[key]
            entries[relative] = FileEntry(
                path=relative, size=len(data), kind="file", sha256=sha256_bytes(data)
            )
        return [entries[name] for name in sorted(entries)]

    async def copy_tree(self, src: str, dst: str, *, ignore: Sequence[str] = ()) -> None:
        """Copy a tree, honouring ignore patterns."""
        source = safe_relpath(src)
        destination = safe_relpath(dst, allow_root=False)
        if await self.exists(destination):
            raise FileExistsError(dst)
        for entry in await self.list_tree(source):
            if entry.kind != "file":
                continue
            parts = entry.path.split("/")
            ignored = any(
                task_copy_path_is_ignored(
                    "/".join(parts[:depth]), is_dir=depth < len(parts), patterns=ignore
                )
                for depth in range(1, len(parts) + 1)
            )
            if ignored:
                continue
            key = f"{source}/{entry.path}" if source else entry.path
            self.files[f"{destination}/{entry.path}"] = self.files[key]

    async def remove(self, path: str) -> None:
        """Remove a file or a tree."""
        key = safe_relpath(path, allow_root=False)
        self.files.pop(key, None)
        for child in self._children(key):
            self.files.pop(child, None)

    async def digest(self, path: str) -> str:
        """Return a file digest or upstream's tree digest."""
        key = safe_relpath(path)
        if key in self.files:
            return sha256_bytes(self.files[key])
        entries = await self.list_tree(key)

        async def read(relative: str) -> bytes:
            return self.files[f"{key}/{relative}" if key else relative]

        return await tree_digest(entries, read)

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 65536,
    ) -> CommandResult:
        """Run the scripted command handler (exit 127 when there is none)."""
        if self.run_handler is None:
            return CommandResult(argv=list(argv), returncode=127, stderr="no command handler")
        result = self.run_handler(argv, cwd, env)
        if inspect.isawaitable(result):
            result = await result
        return result

    def host_path(self, path: str) -> str:
        """Return the workspace path itself (there is no host)."""
        return "/" + safe_relpath(path)

    async def snapshot(self, path: str) -> SnapshotRef:
        """Copy the files under ``path`` into memory."""
        key = safe_relpath(path, allow_root=False)
        snapshot_id = uuid.uuid4().hex
        captured = {k: v for k, v in self.files.items() if k == key or k.startswith(f"{key}/")}
        self._snapshots[snapshot_id] = captured
        digest = hashlib.sha256(repr(sorted(captured.items())).encode()).hexdigest()
        return SnapshotRef(snapshot_id=snapshot_id, path=key, digest=digest)

    async def restore(self, ref: SnapshotRef) -> None:
        """Replace ``ref.path`` with the snapshot's files."""
        await self.remove(ref.path)
        self.files.update(self._snapshots[ref.snapshot_id])


__all__ = [
    "FakeEvaluator",
    "FakeFigureRenderer",
    "FakeFinalEvaluator",
    "FakePaperToolchain",
    "FakeRetriever",
    "FakeTaskGate",
    "MemoryBackend",
    "MemoryWorkspace",
    "completed_result",
    "text_pdf",
]
