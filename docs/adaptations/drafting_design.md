# Task design loop: adaptations and discrepancies

This file records how the task design loop (`drafting/design_models.py`, `drafting/design.py`,
`drafting/graph_design.py`, the five role prompts and the four texts of the shell driver) adapts
upstream AI-Scientist-v3 at commit `df8313889a004f8a44a472e19a833cd547bb828a`, and where the
design of the port and the upstream source disagree. Every change to a prompt text is declared,
with its reason, in `src/langgraph_agentic_scientist/prompts/changes/drafting_design.toml`.

The loop is upstream's phase A1 (`run_design_workflow` in `src/scripts/lib/task_design.sh`, the
contracts of `src/task_drafting/design.py` and `design_cli.py`, and `latent_review.py`).

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/design_task.md` | `prompts/drafting/design_task.md` | A3: one sentence (the design is the structured answer) |
| `src/prompts/validate_task_design.md` | `prompts/drafting/validate_task_design.md` | A3: one sentence |
| `src/prompts/repair_task_design.md` | `prompts/drafting/repair_task_design.md` | A3: one sentence |
| `src/prompts/repair_task_design_structure.md` | `prompts/drafting/repair_task_design_structure.md` | A3: one sentence; "before exiting" becomes "before you answer" |
| `src/prompts/audit_task_design_delta.md` | `prompts/drafting/audit_task_design_delta.md` | A3: one sentence |
| `src/prompts/adjudicate_task_design.md` | none | dormant upstream, vendored only |
| `task_design.sh` lines 846-848 (focused repair verification) | `prompts/workflow/drafting/focused_repair_verification.md` | extracted; the shell variable is `{{TARGET_DIMENSIONS}}` |
| `task_design.sh` lines 856-858 (retry feedback of the judge) | `prompts/workflow/drafting/review_retry_feedback.md` | extracted; `{{DIAGNOSTIC}}` |
| `task_design.sh` lines 1225-1227 (retry feedback of the delta auditor) | `prompts/workflow/drafting/delta_retry_feedback.md` | extracted; `{{DIAGNOSTIC}}` |
| `task_design.sh` lines 1090-1092 (format retry feedback) | `prompts/workflow/drafting/patch_format_retry_feedback.md` | extracted; `{{DIAGNOSTIC}}` |

The five role prompts differ from upstream in exactly the sentences that told an agent to write a
JSON file or to edit a file in place. Every field list, every example and every rule learned from
failures is upstream's text. The shell strings cannot be extracted by `tools/derive_prompts.py`,
which reads Python; they were extracted by hand into
`third_party/ai-scientist-v3/extracted/drafting_design/`, and
`tests/drafting/test_design_prompts.py` re-extracts them from the vendored driver and compares
bytes.

## Discrepancies between the design document and the upstream source

1. **There is no scope rotation in the wired path.** The design lists "rotate repair scope" in
   step 4 of the loop and "scope rotation" among the functions of `design.py`. In the source,
   `rotate_design_repair_scope` runs only when `best_scope_alternatives` is non-empty, and that
   value is produced only by `task_design_scope_alternatives`, which requires exactly three
   reviews. `run_design_consensus` runs one judge and always sets the alternatives to the empty
   string, so the rotation never runs. It belongs to the dormant quorum (omission 13) and is not
   built, with `aggregate_task_design_reviews`, the adjudication request and
   `load_task_design_adjudication`. What upstream does with one judge is the focus rotation
   after two non-improving rounds (`focus_design_repair_payload`), which is carried.
2. **The two retry texts differ.** The design table maps one file, `review_retry_feedback.md`,
   to both `task_design.sh:856-858` and `1225-1227`. The judge's text says "Your previous review
   artifact failed its contract. Correct the artifact and re-check the design."; the delta
   auditor's says "Your previous delta artifact failed its contract. Correct the artifact and
   repeat the audit." The delta text is its own file, `delta_retry_feedback.md`.
3. **The line numbers.** The design cites `task_design.sh:828-848` for the focused section; the
   text is the three strings on lines 846-848.
4. **Retry sections accumulate for the judge and not for the delta auditor.** Upstream appends one
   "# Retry feedback" section to the judge's prompt for every failed attempt (`review_prompt+=`
   inside the loop), so the third attempt carries two. The delta auditor's prompt is rebuilt from
   the base for every attempt (`attempt_prompt="$audit_prompt"`), so it carries only the last
   failure. The patch repairer carries only the last failure as well. All three are carried as
   they are.
5. **Upstream's driver does not stop on three invalid patches at once.** After the third invalid
   patch artifact `run_design_patch_repair` returns the repair's exit status and the whole
   workflow ends; the semantic budget is not spent and the progress file records the failure
   (`repair_failed`, with the non-improving counter raised). The graph does the same: the loop
   ends with the failure `task_design_patch_invalid`.
6. **A review that stays invalid ends the loop.** Upstream returns the status of the third failed
   attempt (`Independent task design judge produced an invalid report`); the graph ends with the
   failure `task_design_review_invalid`. A delta audit that stays invalid does not end the loop:
   the candidate is rejected and the progress records `audit_failure`.
7. **`check_patch` is also the application of the patch.** The design lists `check_patch` and
   `apply_patch` as two nodes. Upstream's `apply-patch` is the check (it validates the review that
   authorises the repair, the patch and the resulting design, and prints the candidate), so one
   node does both.
8. **The legacy review schemas.** `load_task_design_review` also reads schemas 1 and 2, which the
   loop never accepts (`--require-schema-version 3` on every call, and the judge prompt asks for
   3). They are not carried. The delta audit's schema 1 is carried, because the validator
   distinguishes it (`schema_version 1 cannot authorize a repair frontier`).

## Adaptations of the flow

* **A session is a structured role over a read-only directory** (design decisions D1 to D3). Each
  role answers through LangChain structured output with the schema of its record, the
  designer and the structural repairer with `TaskDesign`, the judge with `DesignReview`, the
  patch repairer with `DesignPatch` and the delta auditor with `DeltaAudit`. The host writes the
  file that the prompt names, into a directory per session, next to the files the prompt reads:
  `generate/`, `structure_repair_<n>/`, `review_<round>_a/` (with `parent_review.json`,
  `parent_design.json` and `changed_paths.json` for a focused review), `repair_<n>[_format_<k>]/`
  and `delta_<round>/`. The prompts therefore stay true (`./task.md`, `./task_design.json`,
  `./parent_review.json`, ...). The designer's directory holds `task.md`, so its prompt names
  `./task.md` and `./task_design.json`, where upstream's designer worked in the output directory
  (`./task/task.md`, `./_task_design/task_design.json`).
* **The checks of a role's workspace are vacuous.** Upstream compares `task.md`, the design and the
  reviews before and after each session, validates that the workspace holds only the expected
  entries, and compares the scaffold's tree fingerprint before and after the designer. A
  structured role has a read-only backend (`ReadOnlyBackend`), which cannot change any file, so
  none of these checks has anything to find and none is carried.
* **The designer and the structural repairer have the `file_sha256` tool** (adaptation A7): they
  must write the SHA-256 of `task.md` into the design, and a read-only role has no shell. The other
  roles read only.
* **Retries are graph loops.** A review attempt, a format attempt and a delta attempt are node
  executions of `design_review`, `design_patch_repair` and `design_delta_audit`, each running one
  session, with the counter in state (`review_attempts`, `format_attempts`, `delta_attempts`).
  Structural repairs and semantic repairs have counters of their own (`structure_repairs`,
  `semantic_repairs`).
* **The loop's files are views of state.** `task_design.json`, `accepted_review.json`,
  `design_receipt.json` (upstream's receipt: the SHA-256 of the design file and of the accepted
  review file, and the verdict) and `candidate_ledger.jsonl` are written by the host. The progress
  file, the repair memory, the focus history and the ledger are state
  (`DesignProgress`, `RepairMemory`, `focus_history`, `DesignLedgerEntry`), checkpointed by the
  caller's checkpointer.
* **Failures.** A session that ends without an answer is a failed attempt of its role. For the
  designer it ends the loop (`task_design_generation_failed`); for the structural repairer it is a
  structural repair spent (the design is validated again unchanged); for the judge it is an invalid
  review; for the patch repairer an invalid patch; for the delta auditor an invalid audit.
  Upstream's exit codes 75 (provider capacity) and 78 (billing) are not carried; a transient
  provider failure is retried by `ModelRetryMiddleware` and by the node's retry policy.
* **Pydantic carries the shape and the functions carry the rules.** The records forbid unknown
  fields and use strict integers, so the "fields mismatch" and wrong-type errors of upstream's
  loaders cannot occur for a model answer; the rules that a schema cannot state (the SHA-256 of
  the task, editable paths inside `program_0`, disjoint seeds and feedback, unique values,
  JSON-pointer existence and containment, the verdict arithmetic, repair scope, patch authority,
  the causal test of a regression) are `drafting/design.py` with upstream's messages, because the
  message is the diagnostic that goes back to the role. A design built from a patch goes through
  the same validation, and the one place where a "fields mismatch" can still arise is a patch that
  replaces a whole object with a malformed one.
* **Limits** come from `ScientistConfig.drafting` through the recorded run configuration:
  `design_max_repairs` 8, `design_max_structure_repairs` 3, `design_review_attempts` 3,
  `design_format_attempts` 3, `design_delta_attempts` 3 and `design_non_improving_before_focus` 2.
  A candidate that is rejected after the delta audit spends one semantic repair as every other
  outcome does, so a run makes at most eight patch rounds, each with up to three patch attempts.
* **Capacity.** The input may carry `capacity` (`Capacity`: idea cycles and evaluations per
  attempt, which the designer prompt states as the campaign ceiling); without it the configured
  investigation limits are used.
* **Hooks** run before and after the steps that run a role or publish: the keys are
  `drafting.design_generate`, `drafting.design_structure_repair`, `drafting.design_review`,
  `drafting.design_patch_repair`, `drafting.design_delta_audit` and `drafting.design_publish`. The
  pure checks and routing nodes carry none.

## Not carried

* The three-judge quorum, its adjudication and the scope alternatives that need it (omission 13).
* The workspace and scaffold checks of the shell driver, which a read-only backend makes empty
  (above), the private agent homes and their authentication staging, the progress emitter and the
  log files.
* The re-confirmation of the delta audit inside the latent rebinding command: the audit was
  confirmed when it was accepted, with the same inputs.
