# Paper host side: data, references, validation

This file records how the host side of the paper phase adapts upstream AI-Scientist-v3 at commit
`df8313889a004f8a44a472e19a833cd547bb828a`: the paper data and its grounding ids, the reference
plan and retrieval, the validation gate and the compile, the venue catalog, the watermark, and the
six paper skills. Every change to a prompt or skill text is declared, with its reason, in
`src/langgraph_agentic_scientist/prompts/changes/paper_host.toml`.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/reference_query_plan.md`, `reference_query_plan_edit.md` | `prompts/paper/reference_query_plan.md`, `reference_query_plan_edit.md` | A3: four sentences that told the agent to write the memo and the plan files; the planner returns `{memo, plan}` and the harness writes both files |
| `src/orchestrator/paper_diagnostics.py` (`reference_repair_context`) | `prompts/workflow/paper/reference_repair_context.md` | the fixed text only; the allowed categories and the previous artifacts are added by code |
| `src/agents/skills/write-paper/**` | `skills/write-paper/**` | `SKILL.md` and the templates verbatim; `scripts/validate_claims.py`: the staged-skill directory is added to the directories it excludes (A4) |
| `src/agents/skills/write-paper-standalone/**` | `skills/write-paper-standalone/**` | `SKILL.md` verbatim; `scripts/validate_claims.py` as above |
| `src/agents/skills/validate-paper/**` | `skills/validate-paper/**` | `SKILL.md`: the figure validator's path (A4); `validate_compile_inputs.py`: staged-skill directory added; `validate_figures.py`: imports (see below); the rest verbatim |
| `src/agents/skills/scientific-evidence/**` | `skills/scientific-evidence/**` | `SKILL.md`: six command lines name `.agents/skills/...` (A4); `scientific_evidence.py`: staged-skill directory added; the wrapper verbatim |
| `src/agents/skills/retrieve-references/**` | `skills/retrieve-references/**` | verbatim |
| `src/agents/skills/generate-concept-figure/SKILL.md` | `skills/generate-concept-figure/SKILL.md` | verbatim |
| `src/templates/paper_venues/*.md`, `paper_authors/alife.tex`, `configs/venue/*.yaml` | `venues/*.md`, `venues/authors/alife.tex`, `venues/*.yaml` | verbatim apart from the two path fields of the YAML files (see below) |

## Discrepancies between the design and the source

**The skill scripts are not all unchanged.** The design vendors the scripts unchanged. Two facts
of the source make that wrong for a port that stages skills under `.agents/skills/`:

* `validate_claims.py` (both copies), `validate_compile_inputs.py` and `scientific_evidence.py`
  keep a list of the directories where coding-agent CLIs stage skills, and exclude them when they
  scan a paper workspace for `@evidence[[...]]` links or refuse TeX inputs from them. The
  documentation of the staged `scientific-evidence` skill itself contains evidence-link examples;
  without `(".agents", "skills")` in the list, the scan would report them as dangling links and a
  compile could read a staged style file. One line is added to each list (declared, kind A4). The
  host-side validator has the same list, with the same entry.
* `validate_figures.py` is not self-contained upstream: it imports `read_json`,
  `validate_figure_manifest` and `read_tex_file_bounded` from the upstream repository's own
  `src.orchestrator` modules. In a workspace those modules do not exist. Its two import lines name
  `langgraph_agentic_scientist.paper.validation.figures`, which exports the same three functions
  with upstream's signatures for a host directory. The interpreter that runs agent commands must
  therefore have the package installed (it does for `LocalWorkspace`, whose commands run with the
  host's environment).

**A successful Semantic Scholar retrieval is used.** In `paper.sh` the references are written only
inside the OpenAlex and scholarly branches. When Semantic Scholar returns enough references, no
file exists yet when the scholarly fallback is considered (`reference_outputs_exist` is false),
so the scholarly script runs after a successful retrieval, its result replaces Semantic Scholar's
when it reaches the minimum, and Semantic Scholar's references are written only when both
fallbacks fall short (`downstream fallbacks unavailable; using Semantic Scholar references`). The
fallback's own log line (`OpenAlex fallback did not produce references`) shows that this was not
meant. The port writes Semantic Scholar's references when they reach the minimum and runs the
fallbacks only when they do not, in the order upstream gives them. The Semantic Scholar partial
set is never merged into a fallback's set, as upstream.

**`validation_input_sha256` is here.** The design assigns upstream's `paper_evidence_audit.py` to
the paper loop, but `ValidationReport.fingerprint` (the repeated-failure check of the loop) needs
the hash of the validation inputs. `paper/validation/fingerprint.py` ports
`validation_input_sha256` with its bounds, and the fingerprint of a failure
(`validation_failure_fingerprint`), which the loop uses. Its `paper_input_sha256` also takes
the audit's input paths, so the audit's hashes (`AUDIT_INPUT_PATHS` and the evidence sources,
which are the paper loop's) share its bounds and errors.

**The query planners and the memory protocol.** `reference_query_plan.md` ends with a memory
protocol that asks the agent to append to `./memory.md` and the global memory files. The design
makes the query planners read-only roles, so the planner cannot write them. The prompt text is
unchanged; the planner's decisions are in the plan and its memo.

## How upstream's mechanics are carried

**Validation reads a snapshot, not the file system.** `validate_and_compile_paper` opens the paper
directory with `os.open` and refuses symbolic links at every path component. The port runs on a
`Workspace`, which a caller may implement on a container or a remote sandbox. The validators are
synchronous functions over a `PaperTree` (a read-only view with the same rules: a regular file
whose ancestors are real directories; textual symlink resolution). `WorkspacePaperToolchain`
takes a snapshot of the paper directory once, after the watermark is applied (`SnapshotTree`,
filled from `Workspace.list_tree` and `read_bytes`), and runs the gates on it. The same functions
read a host directory (`DirectoryTree`) for the `validate-paper` skill's script. Files under
`.agents/` and the workspace aliases (`task/`, `results/`, `global_memory/`, `aggregated/`,
`extra_tasks/`) are not read: upstream's aliases are symbolic links that its scan does not enter,
and a workspace without symbolic links may hold copies. Figure PNG and PDF files are read for
their header only; their digests come from the listing entry (the snapshot lists with
`Workspace.list_tree(..., with_digest=True)`).

**Gates run in upstream's order**, the first failure ending the run with upstream's message:
artifacts, watermark, sections, Supplementary Information, BibTeX dependencies and keys,
citations, claims and provenance, evidence bundle, visible language, venue alignment, figure
manifest, compile, rendered Supplementary Information, page budget. `validation_report.json` has
upstream's keys. A failing gate is a `ValidationReport` with `passed=False`, the gate's message
in `errors`, the whole upstream report in `checks`, and a fingerprint of the message and the
inputs. Upstream raises and the shell catches; the port returns. Upstream's hard errors of the
feedback files are `[report.error]`, which is `errors`.

**Compile.** The four commands and their flags are upstream's. They run through
`Workspace.run` with upstream's clean environment built from the host process
(`default_latex_environment`; a caller whose workspace runs elsewhere passes `latex_env`).
Upstream's containerised compile (`container_image`) is the caller's workspace. A compiler that
cannot be started is `PaperValidationInfrastructureError`; `validate` returns it as a failed
report with `checks["infrastructure_error"]` true, as upstream's caller turns every exception into
feedback. The recorder and `.aux` checks read a fresh snapshot of the TeX files after each
command. Approved system TeX roots are checked against the host file system, as upstream does,
and are accepted when the path does not exist there (a container path). Upstream's check that the
verifier's report file is owned by the current user (`st_uid`) has no counterpart through a
workspace and is not ported.

**The evidence verifier is the validator's own copy.** The vendored `scientific_evidence.py` is
written into the paper directory before each run (`.agents/skills/scientific-evidence/scripts/`),
so that an edit of the staged copy cannot change the verdict. It runs through `Workspace.run`
with the configured interpreter (default: the current one) and writes to a temporary file that
replaces `scientific-evidence/verification-report.json` only when its payload is valid and its
exit status agrees with its dangling count.

**The watermark** keeps upstream's text and block (the module is upstream's, and a test compares
it with the vendored original). Upstream has no switch; the toolchain has `watermark=True` and
`options["watermark"]` (the paper phase passes `PaperConfig.watermark`), so that a caller who
publishes into a paper of its own with its own disclosure can switch it off. Switched off, the
validation removes a managed block already in `paper.tex` (one seeded from an earlier paper, or
inserted by the toolchain's own `compile` after a concept figure), so the compiled paper carries
none.

**Venues.** `VenueCatalog` replaces the module-level `lru_cache` catalog and the repository path
constants. `brief` and `author_block` of a venue file name files relative to the directory that
holds the YAML file (upstream named repository paths); `template_dir` stays relative to the
`write-paper` skill, which is staged at `.agents/skills/write-paper`. A caller extends the
catalog with a directory of its own (`VenueCatalog.from_directory(...)` and `with_venues`) and
passes it to the toolchain. The page-budget sentence is `venues/page_budget.toml`, which a test
compares with upstream's function.

**References.** The graph retrieves one query per node execution, so that each query is
checkpointed and a crash repeats none of them. Upstream retried each HTTP request up to five
times, pausing up to 20 s (`Retry-After`); the package keeps no retry loop and no sleep of its
own. `SemanticScholarRetriever` raises `SemanticScholarRateLimited` on 403 and 429 and the node's
`RetryPolicy` (five attempts, 1.5 s doubling) retries the query; a query that still fails is
recorded and skipped, as upstream continues after a failed search. A failure of one candidate's
details skips the candidate. A 403 ends the Semantic Scholar queries (upstream probes the key
first and falls back at once). A candidate whose details failed may be fetched again by a later
query. The retriever receives the accepted paper ids and keys of earlier queries in the query's
`filters`; the retriever itself is stateless. Upstream's built-in catalog of ten optimisation
references is dead code in its retrieval and is not ported. The OpenAlex and scholarly scripts
run through the workspace in the paper directory, as upstream runs them, with upstream's limits
(240 s, 360 s). A retriever that offers `fallback` (`FallbackRetriever`) is asked for the
fallbacks; a caller's retriever without it has none. In repair rounds the planner prompt gets
upstream's repair context, and two identical invalid rounds end the stage (upstream's
`paper_diagnostics reference` returns non-zero and the shell exits). The plan model is lenient
(every field has a default) so that structured output does not refuse a plan that the method would
repair; upstream's rules run on the host.

**Paper data** is built from typed results with upstream's JSON and grounding ids (listed in the
module docstring).

* Programs come from `InvestigationResult.evaluations` (search plane, completed, a finite
  `combined_score`, an evaluation that is not pending); rows are ordered by idea directory,
  attempt directory and program index, as upstream's directory scan orders them.
  `evaluation_evidence.json` is read from the program directory by default, as upstream reads it;
  a caller replaces the reader. What reaches that directory is upstream's: a harness-evaluated
  attempt publishes the evaluator's evidence into `program_1` (upstream's harness evaluation
  leaves the evaluator's results there), while the `evaluate_program` tool of a self-evaluated
  attempt publishes only `metrics.json` and `correct.json`, as upstream's evaluation wrapper
  does, so those rows report the evidence `unavailable`.
* Paths in the data are the paths that processes of the workspace see
  (`Workspace.host_path`), as upstream's absolute paths are.
* The ablation bundle is the variants phase's `output_dir`. Upstream lists `variants/` and
  `extra_tasks/<task>/`; the port reads the variants named by the phase's specs, by
  `pipeline_summary.json` and by the aggregated tables, and the cross-task specs of the variants
  that the aggregated tables name. Listing the directories would hash every file below them.
* The final evaluation is bound from `FinalEvaluationResult`; the verification of its manifest
  (`verify_final_manifest`) is the final-evaluation phase's and is not repeated. The gate on a
  not-qualified outcome is kept (`check_final_outcome_policy`, also applied to a prebuilt paper
  data).
* A downstream study reads the selection record that `DownstreamSelection.selection_path` names;
  the record is verified by the step that wrote it.
* `stage_study_sources` and `verify_study_sources` compare tree digests through
  `Workspace.digest`; empty directories and file modes are not carried (the workspace protocol has
  neither).
* The post-hoc functions (`collect_posthoc_analyses`, `validate_posthoc_ranking`,
  `augment_paper_data_with_posthoc`) are here, with upstream's byte and count bounds, because
  they are part of `paper_data_posthoc.py`; the paper loop calls them.
* `FinalEvaluationMode` becomes "do not pass `final`". Not ported: `prepare_results_dir_for_paper`
  (adaptation 17: the data are built from all workers' records), `build_reference_queries` and
  `generate_results_plot` (no caller upstream), `has_evaluated_programs`,
  `alife_bundle_normalize.py` (a one-off normaliser of one upstream study's bundle, called only
  by a test), `paper_manifest.py`'s file locks (atomic workspace writes serve; the manifest is the
  paper loop's), and the concept-figure transaction of `paper_step.py` (the paper loop's;
  only the prompt-log reader that the figure gate needs is here).
* `paper_data_node` (the default node of the `paper.data` slot) returns `{"failure": Failure}`
  and no `paper_data` when the paper data cannot be built (no evaluated programs, a not-qualified
  final under `qualified_only`).

## Requirements on the rest of the package

* `PaperToolchain.validate` is given `options={"watermark": bool}`; other keys are ignored.
* The default toolchain needs `pypdf` (the page and Supplementary Information checks) and
  `pyyaml` (the venue files) as dependencies of the package, not of an extra.
* `Workspace.list_tree(..., with_digest=True)` must carry the digest of regular files (the
  protocol says it does).
