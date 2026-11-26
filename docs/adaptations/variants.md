# Variants (ablations and baselines): adaptations and discrepancies

This file records how the variants package (`langgraph_agentic_scientist.variants`) adapts the
variants pipeline of AI-Scientist-v3 at commit `df8313889a004f8a44a472e19a833cd547bb828a`
(`src/variants_pipeline/`, `src/scripts/variant_agent.sh`, `configs/variants_config.yaml`), and where
the package's design notes and the upstream source disagree. Every change to a prompt text is
declared, with its reason, in `src/langgraph_agentic_scientist/prompts/changes/variants.toml`.

## Stages and nodes

| Upstream stage | Graph node(s) | Role |
|---|---|---|
| 0 `extract` | `extract` | host |
| 1 `retrofit` | `retrofit_prepare`, `retrofit_agent`, `retrofit_finish` | retrofit agent (R16), no skills |
| 2 `generate` (not with `custom_only`) | `generate` | variant generator (R17), structured, no skills |
| 3 `select` (not with `custom_only`) | `select` | variant reviewer (R18), structured, no skills; fail-open; hard caps |
| (between 3 and 4) custom baselines | `custom_baselines` | host |
| 4 `materialize` | `materialize` | host |
| 5 `execute` | `run_control`, `after_control`, `run_variant` (the `build_run_variant_graph` subgraph: `init_variant`, `edit`, `evaluate`, `scaffold_status`, `verify`, `after_verify`, `next_program`, `finish_variant`), `join_variants` | variant editor (R19, skills `edit-program`, `memory-log`), variant verifier (R20, skills `verify-rubrics`, `memory-log`) |
| 6 `cross_task` | `cross_task` (the `build_cross_task_graph` subgraph) | retrofit agent, then the same variant runs |
| 7 `aggregate` | `aggregate` | host |
| 8 `readme` | `readme` | README summariser (R22), structured, no skills |

The retrofit, the generator, the reviewer and the README summariser run without skills because
upstream passes no `--stage-skills` for them: `retrofit.py`, `generate.py`, `select.py` and
`readme.py` each name a `--skill-step` but never ask for staging, which `variant_agent.sh` does
only with `--stage-skills`. Only `execute.py` asks, for the editor (`variant_edit`) and the verifier
(`variant_verify`). `variant_agent.md` is loaded by no upstream code and is not built.

Each retry is a graph loop with its counter in state, so every agent session is one node execution
with its own checkpoint namespace: the edit loop (`edit_max_tries`, 3), the verify loop
(`verify_max_tries`, 3) and the program loop (`max_evals_per_variant`, 2). The caps, tries and the
session wall clocks (generation 1800 s, retrofit 1800 s, selection 1800 s, README 600 s, 7200 s per
variant and per cross-task adaptation) are `VariantsConfig` and apply to each session.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/variant_retrofit.md` | `prompts/variants/variant_retrofit.md` | none |
| `src/prompts/variant_generate.md`, `variant_select.md`, `variant_readme.md` | `prompts/variants/*.md` | A3: the sentence that names the file the agent writes (one sentence each) |
| `variants_pipeline/execute.py:643-656`, `667-687` (f-strings) | `prompts/variants/variant_edit.md`, `variant_verify.md` | extracted; fields became `{{PROGRAM_INDEX}}`, `{{PREVIOUS_STATUS}}`, ... |
| `materialize.py:470-585` (`_write_variant_task_md`) | `prompts/workflow/variants/variant_task/*.md` (16 pieces) | extracted, one string expression per piece |
| `cross_task.py:657-683`, `699-771` (the two cross-task `task.md` writers) | `prompts/workflow/variants/cross_task_control/*.md` (7), `cross_task_variant/*.md` (13) | extracted |
| `cross_task.py:118-130` | `prompts/workflow/variants/reference_evaluator.md` | extracted |
| `rubric_contract.py` (twelve descriptions) | `prompts/workflow/variants/rubric_descriptions/<id>.md` | extracted |
| `readme.py` (assembly strings, fallbacks) | `prompts/workflow/variants/readme/*.md` (30) | extracted |

`tools/derive_prompts.py` extracts exactly one string expression per entry, so a text that code
assembles from several expressions is carried as one file per expression, and the package
composes the pieces in upstream's order and with upstream's conditions. This differs from the
design notes, which name one `rubric_contract.toml` and one `readme.md`; a single file cannot be
derived from several expressions. The tests execute upstream's own functions (parsed from the
vendored files) and compare their output with the package's: the variant and cross-task `task.md`
files for every branch, the README assembly and its fallbacks, the rubric contract, the rubric
status validators and canonicaliser (400 randomised statuses), the edit and verify prompts, the
CSV schema text, the retrofit prompt and the constants.

Placeholders are rendered in one pass, as everywhere in the package: upstream's sequential
`str.replace` re-expands a placeholder that occurs inside a substituted value (a hypothesis
containing `{{...}}`), the package does not.

Text carried by code and not as package data: the CSV-schema lines of the retrofit prompt
(`retrofit._format_csv_schemas` is two format strings), the numbered plan and component list of
`task.md` (`f"{i + 1}. {s}"`), the `---` separators and blank lines of the README, the
`flat` extra-task default hypothesis `f"{title} ({kind} variant)."` (it uses the piece of the
fallback summary, which has the same text), and the default `program_spec.toml` of a reference
folder (`materialize.DEFAULT_PROGRAM_SPEC`, a file-format literal and not prompt text). The in-code fallback prompts (`_DEFAULT_*`) are not carried: they run
only when a template file is missing, and the registry always has it.

## Adaptations of the flow

1. **No stage selection and no resume from disk.** `--stages`, the `_load_*_from_disk` functions,
   the skipped-stage checks and `_validate_pipeline_summary` are replaced by the checkpointer.
   Every stage is a node; a custom-only run skips stages 2 and 3 as upstream does.
2. **Typed inputs in place of a results directory.** `extract` is given the best program as an
   explicit `ProgramRef`, a downstream selection, or an investigation result (in that order, as
   `resolve_best_program` orders the downstream selection before the best-program manifest). The
   scan fallback for a missing manifest and the results-directory containment check are not
   ported. The reference program is the task's seed, or `baseline_program_override`. The
   manifest keeps upstream's provenance and fingerprint fields with the port's meaning (workspace
   paths, `evaluator_tree_digest` of the task and a tree fingerprint of the best program); it is
   written last, so a bundle without a manifest is an interrupted extraction and is rebuilt, and a
   bundle whose manifest differs from the inputs is refused.
3. **Containers, sandboxes and managed runtimes are the caller's.** `prepare_evaluation_runtime`,
   `resolve_sandbox_image`, the Docker sandbox command, GPU leases, the candidate broker command
   lines, `sandbox_runner.py`, the framed promotion protocol and the requirement `container.enabled`
   are not ported. Evaluation goes through `ctx.evaluator` (plane `variant`). A contracted task is
   evaluated only by an evaluator whose `isolated` is true, unless `allow_unisolated_contracted` is
   set, as upstream refuses a contracted task without container isolation.
4. **The host's canonical analysis is ported, the promotion protocol is not.** After a completed
   evaluation the host writes `analysis/run_summary.csv`, `metric_values.csv` and
   `timeseries.csv` from the evaluator's result (metrics flattened with dots, the evidence's
   candidate time series under `candidate/seed_<seed>/`, the program's own `analysis/timeseries.csv`
   rows found below the evaluator's output directory, validated row by row) with upstream's bounds.
   Output a local evaluator leaves in its output directory is not required to be limited to the
   promotable file names: that check belongs to a sandbox that frames its output.
5. **Evaluation ids.** An evaluation is requested under
   `<run>/variants/<variant>/program_<n>` (`<run>/extra/<task>/<variant>/program_<n>` for a
   cross-task adaptation), derived by the graph, so a resumed node asks again for the same
   evaluation. The CSV and prompts keep upstream's label `<variant>_eval_<n>`. The publication of
   an evaluation writes `.evaluation_pending` first and the manifest last; a crash in between is
   recognised by the marker and the publication is redone. The manifest has the shape of the
   core's `evaluate_program` publication (producer `isolated_evaluator`), which the trusted clone
   accepts.
6. **Evaluator failures.** A timeout is a timeout, an evaluation that did not complete (failed,
   interrupted) ends the variant as upstream's evaluation error does, and an
   `infrastructure_error` (upstream re-raises `CandidateBrokerInfrastructureError` and the
   pipeline dies) is recorded on the run and fails the phase after the join, so the summary is
   still written.
7. **Retrofit.** The agent works on a copy; the sealed `evaluate.py` is restored from the source
   bundle (never touched by the agent) and the evaluator tree digest of the source is re-checked.
   A task runtime directory (`.venv`) is never copied, whether the candidate is brokered or not
   (upstream copies it for a legacy executable). A session that timed out or failed makes the
   retrofit fail, as a non-zero exit does upstream. A contracted `python_rpc_v1:handle` candidate
   is copied unchanged and verified by tree digests (marker schema 2).
8. **Structured roles.** The generator, the reviewer and the summariser answer through structured
   output. The host screens the generator's candidates the way upstream's parser skipped
   malformed entries: an id that is not snake case, a reserved id or a duplicate drops the
   candidate. `kind` is `baseline` or `ablation`. `decision: no_relevant_variants` gives no
   candidates whatever the list holds. The reviewer's failure, timeout or missing answer accept
   every candidate (upstream distinguishes "agent failed", "no selection output" and "parse
   error"; with structured output they are one case). Only the final `variant_selection.json` is
   written; upstream's agent wrote a raw one that the host overwrote.
9. **The program diff** is `diff -ruN reference_program program_0` run in the bundle through
   `Workspace.run` (headers show directory names, not absolute paths), cut at 50,000 characters.
10. **Agent staging.** `active_rubrics.json` and `active_idea.json` (a copy of `variant_spec.json`)
    are staged for the retrofit, edit and verify roles, whose work directory is a bundle. They are
    not staged for the generator, the reviewer and the summariser, whose work directories hold
    nothing else. Cost tracking, container homes and failure logs of `variant_agent.sh` are not
    ported. Skill placeholders are filled with the values `variant_agent.sh` gives them
    (`RESULTS_DIR` and `TASK_DIR` are `.`, `ATTEMPT_ID` is `variant_<id>`, ...).
11. **The editor** has the `clone_program` tool in place of the trusted clone command, as the
    `edit-program` skill says; the verifier has the file tools only.
12. **Verification.** A verifier session that fails counts as upstream's non-zero exit (the next
    prompt carries `verifier exited with code 1`). The host-canonical status, the root mirror and
    the qualification of `overall_pass` in `run_summary.csv` are upstream's. The evidence limit of
    4,096 characters is `VariantsConfig.rubric_evidence_max_chars`.
13. **Cross-task.** The control adapts first, then the other variants run in parallel through
    `Send` (upstream runs them in sequence). Extra tasks are workspace directories; their names
    must be unique and safe, because they are evaluation id components. The `has_adapted_control`
    fallback texts of `_write_variant_cross_task_md` are carried and reachable through
    `write_variant_cross_task_md(has_adapted_control=False)`, but upstream's pipeline never calls
    it that way. As upstream, `expected_effect` and the description appear only when the variant
    has a hypothesis. A folder baseline (`base_program` reference) cannot be adapted: its
    cross-task workspace has no `control_program`, and upstream's `validate_reference_workspace`
    raises there; the port fails that variant instead of the process.
14. **Custom baselines** are named by workspace paths (`VariantsConfig.custom_baselines`); a
    frontmatter `kind` must be `baseline` or `ablation`. YAML frontmatter is read with PyYAML,
    which `langchain-core` already requires.
15. **The README** is written as `README.md` in the output directory, as the code does; upstream's
    documentation calls it `variants_readme.md`.
16. **Events** that upstream sends to `log_event_fn` are `Event` records in the phase's output.
17. **Hooks and slots.** Hooks exist around every step; a hook may update `candidates` (after
    `generate`) and `selection` (after `select` and `custom_baselines`). The variant runner and the
    cross-task subgraph are replaceable through the slots `variants.run_variant` and
    `variants.cross_task` (compiled graphs).

## Discrepancies found while porting

* The design notes say the README summariser and the other three roles get no skills; the source
  confirms it only through the absence of `--stage-skills` (their `--skill-step` is a leftover).
* The design table says the variant `task.md` is carried "verbatim": it is, but as sixteen
  extracted pieces (see above).
* `select.py` passes `--skill-step variant_verify` and `readme.py` too, `retrofit.py` and
  `generate.py` pass `variant_edit`; none of them stages skills.
* Upstream's variants pipeline requires `container.enabled` for every agent stage; the package has
  no such switch.
* `cross_task._finalize_variant` turns the control's kind into `baseline` for the adaptation, which
  makes its rubric contract and `task.md` those of a baseline; the port keeps this.

## What the package does not verify

The package is tested with scripted models and a fake evaluator, and with the local evaluator on a
small task. No run of the whole pipeline with a served model, no container or GPU evaluator, and
no contracted task evaluated through a broker has been exercised.

Performance: `Workspace.list_tree` hashes every file it lists. The evaluator tree digest lists
only the task's top-level entries one by one and never reads omitted or masked ones (`resources/`,
`.venv`, `.git`, `results/`), as upstream does not, but a task directory should keep environments
and large resources outside the tree the variants copy: each variant workspace holds its own copy
of the task files.
