# Review and revision: adaptations and discrepancies

This file records how the AI review (`src/aireview.py` and the pipeline's final review phase) and
the review-driven revision (`src/pipeline/revision*.py`, `src/prompts/paper_revision.md`) of
upstream AI-Scientist-v3 at commit `df8313889a004f8a44a472e19a833cd547bb828a` are carried into
`langgraph_agentic_scientist.review` and `langgraph_agentic_scientist.revision`, and where the
upstream source and the port differ. Every change to a prompt text is declared, with its reason, in
`src/langgraph_agentic_scientist/prompts/changes/review.toml`.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/aireview.py` `REVIEWER_SYSTEM_PROMPT` | `prompts/review/reviewer_system.md` | none (the missing space after "venue." is upstream's) |
| `src/aireview.py` `NEURIPS_FORM` (without the `TEMPLATE_INSTRUCTIONS` it is concatenated with) | `prompts/review/neurips_form.md` | none (its typos and the "Weak Accept" wording are upstream's) |
| `src/aireview.py` `TEMPLATE_INSTRUCTIONS` | `prompts/review/template_instructions.md` | A6: the answer format (see below) |
| `src/aireview.py` `META_REVIEWER_SYSTEM_PROMPT` | `prompts/review/area_chair_system.md` | `{reviewer_count}` is the placeholder `{{REVIEWER_COUNT}}` |
| `src/aireview.py` few-shot paper and review (bz2, Base85) | `prompts/review/fewshot_paper.md`, `fewshot_review.md` | none: the decoded texts; `third_party/ai-scientist-v3/extracted/review/` holds them and a test decodes the upstream constants again and compares |
| `src/aireview.py` `_fewshot_prompt()` f-string | `prompts/review/fewshot_wrapper.md` | fields became `{{PAPER_TEXT}}` and `{{REVIEW}}` |
| `src/aireview.py` `_review_prompt()` f-string | `prompts/review/review_request.md` | field became `{{PAPER_TEXT}}` |
| `src/aireview.py` `_meta_review_prompt()` f-string (one review block) | `prompts/review/meta_review_request.md` | fields became `{{INDEX}}`, `{{COUNT}}`, `{{REVIEW_JSON}}`; code renders one block per reviewer |
| `src/aireview.py` `REVIEW_RETRY_CORRECTION` | `prompts/review/retry_correction.md` | none |
| `src/prompts/paper_revision.md` | `prompts/paper/paper_revision.md` | none |

**A6, the answer format.** Upstream asks for the text `THOUGHT:` followed by a fenced `REVIEW JSON`
block and parses it by hand. The port asks a chat model for a structured answer with two fields,
`thought` and `review` (`ReviewerAnswer`). The format block that names the fences became one
sentence naming the two fields, and the two references to `<THOUGHT>` and `<JSON>` became the field
names; the field list with its descriptions, scales and order, the Decision rule and the sentence
"This JSON will be automatically parsed, so ensure the format is precise." are unchanged.

The review prompts are used exactly as upstream assembles them. The reviewer's user message is the
form, the answer format, the few-shot paper and review and the request that carries the manuscript.
The area chair's is the form, the answer format and one block per review, each review given as the
JSON object of `json.dumps`. A second attempt appends the retry correction to the same message.

## The review graph

`build_review_graph` returns a compiled graph:

```text
record_run_config -> extract_text -> reviewer x N (Send) -> join_reviews -> area_chair
    -> aggregate -> publish
```

Hook step names: `review.extract_text`, `review.area_chair`, `review.aggregate`,
`review.publish`. The reviewer tasks are targets of `Send` and carry no hook nodes.

**Carried exactly.** The text extraction chain (`pymupdf4llm`, then `pymupdf`, then `pypdf`; the first
text of at least 100 characters; an extractor that raises counts as empty); the checks of the PDF
(`.pdf` extension, `%PDF-` header, at most 100 MiB); five reviewers and one area chair; at most two
attempts for each call, the second with the retry correction and no validation error added; the
validation of a review (`aireview._validate_review`: every field required, the nine score ranges,
booleans refused as numbers, `Decision` in `Accept`/`Reject`, `Ethical Concerns` a boolean, list
fields of strings), enforced by `AIReview` with upstream's JSON keys as field aliases, so that
`ai_review.json` has upstream's keys; the nine scores of the area chair's answer replaced by the
rounded mean of the reviewers' (Python's `round`, which rounds halves to the even integer, as the
NumPy-based upstream does); `temperature=1`, `seed=attempt` and the completion token limit (16,384),
bound to each call with `Runnable.bind`; the artifacts `<stem>.json` and `<stem>.log` with the
latest-review copies `ai_review.json` and `ai_review.log`, the failed review clearing the JSON copy and
writing `[ai review failure]` to the log, and the status in `latest_manifest.json` under the
manifest key and under `ai_review`; a failed review never discards the paper.

**Adapted.**

- *Any chat model, structured output.* Upstream calls one OpenAI model (`o4-mini`) with a plain
  completion and parses JSON from the text. The port calls the models of the `ai_reviewer` and
  `area_chair` roles through `with_structured_output`, with the method that the model registry gives
  each role. An answer that is missing or violates the review contract is invalid and is asked for
  again. Scores are therefore not comparable with upstream's.
- *The five reviewers run in parallel*, not one after the other. The calls are identical and
  independent. A reviewer with two invalid answers is a failure of the review, as it is upstream;
  the other reviewers' calls are made.
- *Transport failures* are retried by the node's `RetryPolicy` (six attempts, exponential backoff
  of 1, 2, 4, 8 and 16 seconds with jitter) instead of `backoff.on_exception`. The policy retries
  the transient provider failures that `is_transient_provider_error` recognises (connection errors,
  timeouts, HTTP 408, 409, 425, 429 and server errors), which is wider than upstream's three OpenAI
  error classes. A failure that is not transient is not retried. After the last attempt the node's
  error handler records the failure, so the review fails and the run goes on, as upstream's review
  phase is degraded without failing the pipeline. The number of attempts is that of the compiled
  policy (`reviewer_retry_policy`, an argument of `build_review_graph`), not a setting of the
  configuration, because a compiled policy cannot read the run's configuration.
- *Consent.* As upstream, selecting the review phase is the consent to send the manuscript to the
  reviewer models. The graph asks nothing.
- *`review.fewshot`* leaves the example paper and review (about 60,000 characters) out of the
  reviewer's message, for models with a small context. Upstream always includes them.
- *`review.sampling`* set to `none` leaves temperature, seed and the token limit to the model, for
  providers that reject them.
- *The text is in state*, and may be given as `paper_text`, in which case the PDF is not read. The
  extraction runs in a worker thread (`asyncio.to_thread`) because the libraries block.
- *The manifest* is updated when `latest_manifest.json` exists in the paper directory and is left
  alone when it does not, so that a manuscript can be reviewed outside a pipeline. Upstream also
  requires the manifest's `paper_pdf` to name the reviewed PDF; the port does not check this, and
  paths in the manifest are workspace paths. An unusable manifest fails the review.
- *The reviewer's `thought`* is asked for and not kept, as upstream does not keep it. It is optional
  in the answer.
- *Scores* are stored as numbers; integral values are written as integers again, so a review file
  looks as upstream's does.

**Not ported.** The cost accounting of the review calls (`ai_review_cost.py`); the subprocess,
`uv run --script`, the restricted environment and the reading of an `.env` file (`final_ai_review.py`,
`agsci/review.py`), because credentials belong to the chat model that the caller supplies; the
wall clock of the managed subprocess (`review.timeout_seconds`, 1800 s, and its grace period), because
the calls are made by the caller's model clients, which carry their own timeouts, and a caller can bound
a run in the run configuration; the publication lock and the no-follow file operations, which the
workspace's safe paths replace; the `agsci review` command, because the package has no command
line (the graph is the standalone reviewer: `build_review_graph().ainvoke({"pdf_path": ...})`).

## The revision graph

`build_revision_graph` returns a compiled graph:

```text
record_run_config -> check_preconditions -> stage_revision -> paper -> retention_check
    -> snapshot_original -> promote -> second_review -> finalize
```

Hook step names are `revision.<node>` for each node after `record_run_config`.

| Upstream (`pipeline/revision.py` and the modules it uses) | Node |
|---|---|
| `_initial_review_path`, `load_revision_feedback`, the manifest check, the replay of a promoted revision | `check_preconditions` |
| `_next_attempt_dir`, `_reusable_attempt`, the fixed options of `build_revision_command` | `stage_revision` |
| the call of `paper_edit.sh` with `--external-review` | `paper` (slot `revision.paper`) |
| `validate_revision_evidence`, `_validate_revision_workspace` | `retention_check` |
| `promote_revision_workspace` (snapshot, archive, overlay, manifest, check) | `snapshot_original`, `promote` |
| `_run_after_revision_review`, `_record_rereview_status` | `second_review` (slot `review`), `finalize` |

**Carried exactly.** A completed first review is required: the named file
`ai_review.before_revision.json`, or the latest `ai_review.json`, which is then copied to the named
file together with its log and recorded under `before_revision`; a replay must keep the named review
(`initial_review_must_be_preserved`). The review must be usable as feedback (at most 1 MiB, an
object, `Summary` a string, four list fields of strings, at most 100 items each). The fixed options
of the edit are upstream's: `reference_mode` is `preserve`, `figure_refresh_mode` is `auto`,
`skip_postdraft_enrichment` is true. As upstream's command passes the post-hoc directory and the
final evaluation only when the paper being revised has them in its `paper_data.json`, the graph
hands `posthoc` and `final` to the paper graph only in that case. The retention check is upstream's:
the original's `paper_data.json` must be contained in the revised one after volatile timestamps,
absolute locations and the editorial annotations of post-hoc analyses are set aside, with the
list matching by specificity; the revision must have a method-scope audit, a passing evidence audit
and a validation report that passed and records that audit. Publication keeps the original under
`revisions/revision_01/before/` (the original's files, `original_manifest.json`, the first review,
the figures and `snapshot_complete.json`, created once); original-only audited inputs go to
`before/ancillary/` with their relative paths; the revision's files, figures and audit-bound inputs
are copied over the paper; the manifest of the revision has its paths moved into the paper
directory and gains the `revision` and `before_revision` entries; the published paper is checked
against the snapshot. A promoted revision is not edited again: the graph checks it and runs only
the second review if that is missing. A saved attempt that still passes the checks is published
instead of being edited again. A failed edit, evidence check or publication retains the original;
a failed second review keeps the revision and degrades the phase.

**Adapted.**

- *The paper graph is a slot.* `revision.paper` receives `RevisionPaperInput`: `mode`
  `revision`, the published directory as `previous_paper_dir`, the attempt directory as
  `output_dir`, the first review as `external_review`, the fixed `edit_options`, the normalised
  `revision_feedback` and what the paper graph is otherwise given (`task`, `run_id`, the results
  of earlier phases or a prebuilt `paper_data`). It returns a `PaperResult` whose `paper_dir` is the
  revised paper. The default is the package's paper graph. The slot is invoked by a wrapper node,
  so that the paper graph's result is not merged into the revision's state.
- *The feedback files* (`revision_feedback.json` and `.md`), the snapshot `previous_paper/` and the
  restoration of the references after the edit are steps of the paper edit in upstream
  (`paper_edit.sh`), and the revision graph does not write them. It validates and normalises the review
  (`load_revision_feedback`) and gives the result to the paper graph as `revision_feedback`; the
  renderings of upstream's two files are `render_revision_feedback_json` and
  `render_revision_feedback_markdown`, and `write_feedback_files` writes them.
- *Publication without an atomic exchange.* Upstream stages a copy of the paper, builds the result
  in it and exchanges the directories atomically, with a journal for recovery. The workspace
  protocol has no atomic exchange; the graph snapshots the paper (`snapshot_original`), makes
  the same changes in place and restores the snapshot if anything fails, so the original is
  retained exactly. A resume re-enters `promote`, which first restores the snapshot when an earlier
  attempt left partial work. The journal, the portable publication protocol and the descriptor-level
  copying are not ported.
- *The original's audited inputs* that are archived are those recorded in the original's
  `paper_evidence_audit.json` (`input_sha256`); upstream recomputes them from the staged copy.
- *Two checks belong to the paper package.* The validity of the method-scope audit and the
  recomputation of the evidence audit's input hashes are the paper package's
  (`paper.validation.scope_audit.validate_method_scope_audit_in` and
  `paper.audit.audit_input_sha256`), and those are the defaults. A caller can replace them with the
  slot functions `revision.scope_audit` and `revision.audit_input_hashes`
  (`async (directory, runtime)`). When the paper package has neither and the caller supplies
  none, the check covers what the recorded files say (the scope audit's presence, the audit and the
  report passing, and the hashes of the audit and of the report agreeing), and
  `RetentionReport.skipped_checks` and `PromotionRecord.skipped_checks` name what was not made.
- *Validation report.* The report's `ok` is read as upstream does; `passed` is accepted as well.
- *Paths in manifests* are workspace paths. Manifests that hold absolute host paths are understood,
  and a rebased path keeps the form it had.
- *The manifest's `revision` entry* holds workspace paths (`source_dir`, `original_snapshot`,
  `before_review`).
- *The review phases keep upstream's names.* The first review of a revision is written with the stem
  `ai_review.before_revision` and the key `before_revision`, the second with `ai_review.after_revision`
  and `after_revision`; the review graph takes both as input.
- *Not ported.* The revision command line and its forwarding of paper settings; the exclusion of
  `paper_review_rounds` and `contributions_file` from it (upstream runs the edit without the
  paper's own review rounds: the paper graph in revision mode is expected to do the same); the
  workspace aliases of the published tree (`ablations`, `results`, `task`, ... symlinks), because the
  port builds no merged directory; the output tail and failure event formatting; the publication
  locks, the pinned directory descriptors and the portable publication journal.

## Notes on how upstream divides the work

- Upstream's `revision.py` keeps five fields of the review (`_FEEDBACK_FIELDS`) and uses them only to
  validate the review. The sixteen-field feedback, with `untrusted_reviewer_data` and `source_sha256`,
  is written by `paper_edit.sh` (`stage_external_review_feedback`); both are in `feedback.py`.
- The `paper_edit.sh` check on the review (at most 100 items per list) and the `revision.py` check
  (types) are separate in upstream; the port applies both before the edit.
- Upstream's `review_text` and `load_paper` each require 100 characters; the port requires them of
  extracted text and of text given by the caller.
- `final_ai_review` requires the manifest's `paper_pdf` to name the reviewed PDF; see above.
