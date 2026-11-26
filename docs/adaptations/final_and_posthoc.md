# Sealed final evaluation and post-hoc analysis: adaptations and discrepancies

This file records how the final evaluation graph (`final_evaluation/`) and the post-hoc graph
(`posthoc/`) adapt upstream AI-Scientist-v3 at commit `df8313889a004f8a44a472e19a833cd547bb828a`
(`src/final_evaluation/`, `src/posthoc_pipeline/`, `src/scripts/posthoc_agent.sh`,
`configs/posthoc_config.yaml`, `docs/posthoc_pipeline.md`, `docs/evaluation_evidence.md`,
`docs/paper_final_outcomes.md`), and where the port's design notes and the upstream source
disagree. Every change to a prompt text is declared, with its reason, in
`src/langgraph_agentic_scientist/prompts/changes/posthoc.toml`.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/posthoc_bootstrap.md` | `prompts/posthoc/posthoc_bootstrap.md` | none |
| `src/prompts/posthoc_agent.md` | `prompts/posthoc/posthoc_agent.md` | none |
| `src/prompts/posthoc_generate.md` | `prompts/posthoc/posthoc_generate.md` | A3: line 3 asks for the structured response, which the harness writes to `posthoc_bundle.json` |
| `src/prompts/posthoc_select.md` | `prompts/posthoc/posthoc_select.md` | A3: line 1 asks for the structured response, which the harness writes to `posthoc_selection.json` |
| `src/posthoc_pipeline/materialize.py` lines 173-211 (the generated analysis `task.md`) | `prompts/workflow/posthoc/analysis_task/*.md` | extracted string by string; the bullet and numbered list items and the joiners stay in code (`posthoc/materialize.py`) |
| `src/posthoc_pipeline/rubric_contract.py` lines 13-29 | `posthoc/rubric_contract.py` constants | verbatim; extracted (`dict_value`) and pinned to the constants in code |

The in-code fallback prompts of upstream (`_DEFAULT_BOOTSTRAP_PROMPT`, `_DEFAULT_GENERATION_PROMPT`,
`_DEFAULT_SELECTION_PROMPT`, `_DEFAULT_POSTHOC_AGENT_PROMPT`) run only when the template file is
missing and are not carried.

Tests compare the port with the vendored upstream source: the two verbatim prompts byte for byte,
the two structured prompts line by line (one line differs), the rubric ids, descriptions,
contract and status validation by running upstream's `rubric_contract.py`, and the generated
`task.md` by running upstream's `_write_analysis_task_md` on the same specifications.

### Texts kept in code

The rubric descriptions (values of the dictionary `_RUBRIC_DESCRIPTIONS`) and the formats of the
list items of the generated `task.md` (`` - `{item}` `` and `{index + 1}. {step}`, f-strings inside
generator expressions) are extracted by `tools/derive_prompts.py` (kinds `dict_value` and
`fstring`) into `third_party/ai-scientist-v3/extracted/posthoc/`. The port keeps them as constants
and format strings in code; a `[[pin]]` entry in `posthoc.toml` extracts each one from the package
source too and `--check` fails when the two differ.

## Sealed final evaluation

### What is kept

* **Once only.** The evaluator owns it: for an evaluation id that has a recorded result it returns
  that result and never evaluates again; for an id that started and whose result cannot be
  obtained it raises `FinalEvaluationAlreadyStarted` (upstream: `invocation.json` without a
  manifest). The graph never calls the evaluator under a second id and never reruns a failed
  phase: `FinalEvaluationAlreadyStarted` ends the phase `failed` with the upstream message
  ("final evaluation already started; refusing rerun") and the category
  `final_evaluation_already_started`.
* **Resolve the frozen incumbent** (`resolve_frozen_incumbent`): a selected downstream program
  first, else the investigation's best program, which must lie inside the investigation's results
  directory and have a safe entrypoint. An explicit `incumbent` input wins over both, for callers
  that run the phase alone.
* **Freeze.** The digests of the program (upstream's tree digest), of the task scaffold (the
  fingerprint of a contract, else the tree) and of the final evaluator file bind the evaluation.
  The `FreezeRecord` is in state when the node completes.
* **Verification** (`verify_final_manifest`), applied to the result the evaluator returns, whatever
  evaluator it is: the field rules of manifest schemas 1, 2 and 3 (through the core's
  `validate_final_manifest`: exact fields, `correct` boolean, schema 1 only for correct programs,
  the outcome matching `correct`, the declared final metrics exactly the manifest's, all finite, the
  digest format, a completed compute usage for schema 3), then: the program and the task scaffold
  unchanged since the freeze ("frozen incumbent changed during final evaluation", "task scaffold
  changed during final evaluation"), the result bound to the frozen task, evaluator and program
  digests, with a contract the declared final metrics and the compute limits equal to the
  contract's, and, when the evaluator kept its results where the workspace can read them
  (`output_dir/results`), the digest of the results. A failure ends the phase `failed` with the
  category `final_manifest_invalid` and no result.
* **Infrastructure versus candidate failures** keep upstream's split: `FinalEvaluationInfrastructureError`
  becomes the scope `infrastructure` (upstream exit 4), any other `FinalEvaluationError` the scope
  `method` (exit 1).
* **Contracted tasks refuse an evaluator that does not isolate the candidate**
  ("contracted final evaluation requires container isolation"), unless
  `allow_unisolated_contracted` is set; the exception is recorded as an event.

### The drawn inputs

`draw_inputs` is a node of its own between `freeze` and `evaluate_sealed`, and a replaceable slot
(`final_evaluation.draw_inputs`; a compiled graph or an async callable
`(DrawInputsInput, runtime) -> DrawInputsOutput`). The default returns `None`: the task's own
`resources/final/**`. A completed draw is checkpointed, so it runs once even when `evaluate_sealed`
is interrupted (a `DeferredFinalEvaluator`) or crashes and the run is resumed (both are tested).
The result goes through a routing node `attach_inputs` into `FreezeRecord.sealed_inputs`, which is
what the evaluator receives in the request. `evaluate_sealed` does nothing with an effect before
its single call of the evaluator, under the id `<run>/final`.

### What is not ported

The broker and everything that exists for it belong to the caller's `FinalEvaluator`: container
launch, the three launch attempts with a one and a two second backoff, image re-resolution,
recovery of retained broker containers, GPU leases and quarantine, the reset-transcript check
against `protocol.final_reset_sequence`, the RPC limit and the wall clock enforced by the broker,
read-only sealing of the results, `final_launch_failure.json` and `final_cleanup_failure.json`.
Upstream has no setting for the launch attempts either (three is a constant of its runner, applied
to brokered launches only), so no upstream configuration carries one and the package has none; a
`FinalEvaluator` that retries launches owns that number itself. The deadline is kept:
`final_evaluation.timeout_seconds` is upstream's `final.timeout_seconds` (21600), passed in the
request, and the local evaluator bounds the evaluation by the smaller of it and the contract's final
wall clock, as upstream's runner does.

The graph does not write `final_manifest.json` or `invocation.json`: the evaluator does (the local
evaluator writes both), and a caller whose evaluator keeps no manifest still gets the verified
`FinalEvaluationResult` in state.

### Discrepancies with the design notes

* The result of the phase is the state key `final` (a `FinalEvaluationResult`), next to `frozen`,
  `final_status` (`completed` or `failed`) and `failure`; the pipeline maps these to its phase
  status and the `final_outcome_policy`.
* The design left it open how the drawn inputs enter the `FreezeRecord`; the slot returns them in
  `sealed_inputs` and a routing node attaches them.

## Post-hoc analysis

### The flow

`extract` -> `bootstrap_prepare` -> (`bootstrap_reset` -> `bootstrap_agent` -> `bootstrap_evaluate`
-> `bootstrap_record`) up to `bootstrap_max_attempts` times -> `bootstrap_finish` ->
`restore_evidence` -> `generate` -> `select` -> `apply_constraints` -> `materialize` -> one
`run_analysis` subgraph per analysis (`analysis_begin`, then per attempt `attempt_prepare`,
`attempt_execute`, `attempt_inspect`, then `analysis_finish`) -> `publish`. Every step has hook
nodes `posthoc.<step>.before` and `.after`; the subgraph's steps are `posthoc.analysis.<step>`.
Hooks of `generate`, `apply_constraints` may update `candidates` and `selection`.

All upstream rules are carried: the bundle contents, the evidence allowlist and its bounds, the
checkpoint candidates and their validation, the primary metric inference and guardrail, the
attempt limits, the six-rubric contract and its status normalisation, the hard constraints of the
selection, the attempt inspection (required root files, rubric status, manifest keys, evidence
paths, expected outputs, the pristine `task_bundle`, `reran_evaluator`, the checkpoint), the
timeout ending the loop, the publication files and their keys (`ranking.json`,
`latest_manifest.json`, `pipeline_summary.json`).

### Roles

R23 (bootstrap), R24 (generator) and R25 (selector) get no skills and no package tools. The
generator and the selector are structured, read-only roles over `generation/`; the host writes
`posthoc_bundle.json`, `change_summary.json` and `posthoc_selection.json`. R26 (executor) gets
the `run` skill set (`edit-program`, `evaluate-program`, `verify-rubrics`, `memory-log`), the
`clone_program` tool and an `evaluate_program` tool whose budget is 0: upstream's
`posthoc_agent.sh` defaults to one evaluation, but `execute.py` passes `--max-program-evals 0`,
and the analysis forbids evaluating. Its working directory is the attempt directory, and the
active files `posthoc_agent.sh` stages (`active_idea.json`, `active_rubrics.json`) and the
placeholder values of the skills (`ATTEMPT_ID`, `IDEA_ID` as the analysis id, `TASK_DIR` as
`task_bundle`, ...) are staged the same way.

### Adaptations

* **Evaluation goes through `ctx.evaluator`.** Upstream's bootstrap reran the task's `evaluate.py`
  with `--results_dir` set to `program_0`, so the evaluator's outputs (and a checkpoint a program
  saves next to them) landed in the program. The evaluator of the port keeps its outputs in its
  own area. The host therefore promotes them after the call: `metrics.json` and `correct.json`
  from the validated result, and every file of the result's `output_dir` when the workspace can
  read it. An evaluator that keeps its outputs where the workspace cannot read them still gives
  the metrics and correctness, but a checkpoint the evaluation saves is not found, and the
  bootstrap then degrades. The evaluation id is `<run>/posthoc/bootstrap_<n>`, plane `bootstrap`.
* **The bootstrap bundle holds the search evaluator**, as upstream's does, because the prompt
  carries its source and `_validate_bundle` requires it; the evaluation itself uses the task's own
  copy through `ctx.evaluator`, so editing the bundle's copy changes nothing.
* **Sealed files stay out.** For a task with a contract the bundle leaves out the sealed final
  evaluator, the final resources and `task_contract.json`; no agent of the search side may read
  them. Upstream relied on the container's masks.
* **Symlinks are not carried** (the workspace protocol has no operation that creates them); staged
  evidence is not made read-only; the manifest verification and `restore_verified_evidence` serve
  instead and are tested.
* **Checkpoint validation reads files through the workspace**, which has no ranged read, so a
  checkpoint larger than 4 GiB is accepted after the non-empty check only, and `.npy` files are
  checked by their header when `numpy` is not installed (upstream imports it).
* **Name hints of a checkpoint and the paths of `_results.json` are read relative to the program.**
  Upstream applied the `checkpoint`, `ckpt` and `weights` hints to the absolute path, so a program
  below a directory of that name made every file of a checkpoint suffix a candidate; a path that
  `_results.json` names through a linked directory or with `..` components is found only by the
  name scan.
* **Expected outputs and required artifacts follow the workspace path rules**: a `..` component,
  a backslash, a leading `~`, or leading or trailing whitespace makes a path missing (upstream
  resolved such a path inside the directory), and a path through a link that leaves the workspace
  is missing instead of an error.
* **Analyses run in parallel** through `Send`, each with its own attempt loop and agent sessions
  (upstream: one after another). Bound it with the run config's `max_concurrency`; use 1 for
  analyses that use a single GPU.
* **Selection is two steps** (`select`, `apply_constraints`) so that a hook can see the selector's
  answer, and fails open exactly where upstream does: no answer accepts everything
  ("accepted (selector unavailable)"), a candidate the selector did not review is accepted
  ("accepted (not reviewed by selector)"). A selector schema mismatch has no counterpart, because
  the answer is structured.
* **Extraction errors, a missing baseline metric and an unresolvable primary metric end the phase
  `failed`** (upstream raised; the pipeline treated the exit as a degraded optional phase). A metric
  that cannot be inferred from the evaluation of a bootstrap attempt fails that attempt instead of
  crashing the phase.
* **The phase status.** `completed` only when at least one analysis is valid; `degraded` when none
  is (failure category `no_valid_analyses`, or `bootstrap_not_ready` when the bootstrap also
  degraded; upstream's standalone exit code 1 for the latter); `failed` when the inputs could not be
  extracted.
* **Integrated mode.** Upstream's pipeline runs the phase with `checkpoint_mode='optional'` and
  `retrain_if_checkpoint_missing=false` (artifact only). The input flag `artifact_only` selects
  that mode; the configuration's defaults are the stage defaults (`required`, `true`).
* **The reference program** is `task.seed_program` (default `task/program_0`); upstream took the
  retained baseline of a downstream selection (`baseline_program_dir`), which the port's
  `DownstreamSelection` does not carry. They differ only if the selection retained another
  baseline.
* **Paths in prompts and records are workspace paths** (upstream's were absolute host paths), for
  example `analysis_dir` in the specification the executor reads and in `attempt_result.json`.
* **Where the evidence comes from.** The variants phase's output directory (when its status is
  `completed`) or an explicit `evidence_dir`. The layout is upstream's: `pipeline_summary.json`
  (schema 1, `total_variants`, `total_completed`, `variants[].variant_id` and `status`), `README.md`,
  `generation/variant_bundle.json`, `generation/variant_selection.json`,
  `aggregated/<task>/{run_summary,metric_values,timeseries}.csv` and
  `variants/<variant>/task_bundle/program_<n>/evaluation_evidence.json`. The variants phase must
  write that layout for the evidence to be staged.
* **The stable pointer.** Upstream copies `latest_manifest.json` to the parent of the output
  directory. The port does so when the parent is a directory of the workspace and leaves the
  workspace root alone, where the file would collide with the paper's.
* **The analysis workspace of a stale analysis** is removed when the candidates are materialised
  again, by candidate id; upstream removed every directory below `analyses/` that was not
  accepted.

### Not carried

The prompt files staged for the agent CLI (`bootstrap_prompt.md`, `posthoc_generate_prompt.md`,
`posthoc_select_prompt.md`, `posthoc_attempt_prompt.md`: the prompt is the session's first
message), the agent logs and process reports (`bootstrap_attempts/attempt_NN/*.log`,
`posthoc_agent_*.log`), the event log and the progress summary, the stage selection of the CLI
(`--stages`, resuming from stage numbers: the checkpointer resumes at the step that was running),
`integration_status.json`, `analysis_timeout_grace_seconds` (an `asyncio` timeout has no process
group to wait for), and the agent, container and logging settings of `posthoc_config.yaml`.

### Notes on upstream behaviour kept as it is

* `checkpoint_guardrail_respected` still describes an evaluator rerun, which the analysis forbids.
* After a failed bootstrap the bundle is reset and the report's primary metric and guardrail are
  those of the reset bundle, not of the failed attempts; the attempts' own values are in
  `retry_attempts`.
* `expected_outputs` of the generator that are prose become safe relative paths (`outputs/01_...`);
  a candidate whose `id` is not a safe directory name is skipped.
* A timed-out session ends the analysis as `timeout` even when its outputs would validate.
