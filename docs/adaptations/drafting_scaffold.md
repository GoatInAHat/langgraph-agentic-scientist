# Task scaffold drafting: adaptations and discrepancies

This file records how the scaffold half of task drafting (upstream's `src/scripts/task.sh` and its
helper modules, from the first draft of the scaffold to its publication) is carried into the
package, and where the upstream source behaves differently from what a reader of its documentation
would expect. It covers `langgraph_agentic_scientist.drafting.graph_scaffold`, `.graph`, `.gate`,
`.scaffold`, `.qualification`, `.history`, `.contract_lint` and `.models`, the five role prompts,
the workflow texts and the `draft-task` skill. Every change to a prompt or skill text is declared,
with its reason, in `src/langgraph_agentic_scientist/prompts/changes/drafting_scaffold.toml`;
the design loop that precedes the scaffold loop is documented in `drafting_design.md`.

## The flow, step by step

| Upstream | Graph step(s) | Upstream source |
|---|---|---|
| copy of the task into the drafting directory | `prepare` | `task.sh` (`ensure_staged_task_copy`) |
| A2 draft, `DRAFT_MAX_ATTEMPTS` (3) | `draft`, `inspect_scaffold` | `task.sh` (`draft_scaffold_until_complete`, `inspect_staged_scaffold`) |
| canonical `capture.py`, `program_spec.toml`, entrypoint | `canonicalize` | `task.sh` (`canonicalize_staged_scaffold`, `validate_staged_paths`) |
| A3 contract preflight, `CONTRACT_MAX_REPAIRS` (3) | `preflight`, `preflight_repair`, `close_preflight_repair` | `task.sh` (`repair_contract_until_valid`) |
| A4 smoke, `SMOKE_MAX_REPAIRS` (3), `SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS` (3) | `smoke`, `open_smoke_transaction`, `smoke_repair`, `close_smoke_repair`, `smoke_accepted` | `task.sh` (`smoke_until_valid`) |
| A5 alignment, `ALIGNMENT_MAX_REPAIRS` (10), at most 2 * (10 + 4 + 1) judge calls | `alignment_view`, `alignment_judge`, `check_alignment`, `alignment_passed`, `alignment_failed`, `open_alignment_transaction`, `alignment_repair`, `close_alignment_repair`, `alignment_validation_failed`, `alignment_boundary_retry` | `task.sh` (the review loop, `repair_alignment_transactionally`) |
| A6 qualification, `QUALIFICATION_MAX_REPAIRS` (4), `QUALIFICATION_MAX_INFRASTRUCTURE_ATTEMPTS` (3) | `qualify`, `qualification_passed`, `qualification_failed`, `open_qualification_transaction`, `qualification_repair`, `close_qualification_repair`, `qualification_validation_failed`, `check_qualification_candidate`, `validate_qualification`, `commit_qualification_repair`, `qualification_exhausted` | `task.sh`, `lib/task_qualification.sh`, `task_drafting/qualification*.py` |
| A7 publication | `publish` | `task_drafting/publication.py`, `task.sh` |

Steps that run a role, a gate, an evaluation or an effect carry hook nodes under the keys
`drafting.<step>` (`SCAFFOLD_STEPS`). Bookkeeping nodes that only route or count carry none.

**The order of the qualification repair.** Upstream runs the repaired scaffold through the
contract and smoke gates, then through the full production qualification, and only then back
through the judge; the judge sees the calibration record, and a successful qualification is
reused (not run again) when the judge accepts. The graph follows that order
(`check_qualification_candidate` creates the calibration and leads to `qualify`, whose success
leads back to `alignment_view`). A judge that rejects a pending repair records the repaired
fingerprint in the failure history, retains the rejected edits as history, and returns to the
qualification repair loop with the repair's source report.

**Which repairs are scoped.** Only smoke repairs are scoped: a snapshot of every file and of the
parsed contract is taken before the first repair of a chain, a repair may change only the two
evaluators, `requirements.txt`, the baseline entrypoint and (by policy) the contract's declared
digests, and a change outside that boundary is undone. Contract-preflight repairs edit the staged
scaffold directly, with no boundary beyond the safety checks on the staged paths; alignment and
qualification repairs are tree transactions (the whole authored tree is restored on rollback) with
no mutation boundary. A repair that changes the canonical `task.md` is refused in every case.

**A blocker is a failure.** An agent that calls `report_blocker`, a session that fails, and a
session that times out end the repair as upstream's non-zero exit did: the pending transaction is
restored and drafting fails with the blocker as its detail. The drafter's blocker only matters when
the scaffold is still incomplete after its last attempt.

## Where the roles work

The drafting directory is `<run>/_task_drafting/`, beside the task snapshot `<run>/task`.

* The drafter works in the drafting directory. Upstream stages the `task` skill set
  (`draft-task`, `memory-log`) in that directory once, at the start, so only the roles whose working
  directory it is see them. The package stages them under `.agents/skills/` there, rendered with
  the drafter's placeholder values, for the drafter only. The repair roles work in the staged task
  directory (`<run>/_task_drafting/task`) and the judge in a separate view, so none of them sees the
  skills. (Upstream's task designer also works in the drafting directory, which is the concern of
  the design loop.)
* The staged copy of the task is `<run>/_task_drafting/task`; the accepted design is
  `<run>/_task_drafting/_task_design/task_design.json`, where the drafter's prompt says it is. In
  the legacy workflow there is no design and the prompt still names that path (upstream does the
  same); the judge is then pointed at `task.md`.
* `<run>/_task_drafting/.harness/` holds host-only files outside every agent's root: the judge's
  view of each round (`judge_view_<n>`, the staged scaffold without runtime state, with the design
  as `_accepted_task_design.json`) and the authorisation of a dependency removal.
* Prompts are rendered as upstream renders them: paths with `str(Path(...))` semantics (the
  drafter's `TASK_DIR` is `task`, a repair prompt's is `.`), every prompt stripped of trailing
  newlines (upstream builds them with command substitution), retry sections appended with the
  exact newlines upstream's code prints, and the gate text appended to every repair prompt.

## Limits, and where each is tested

Every limit is read from the recorded configuration (`ScientistConfig.drafting`) and is a tested
bound (`tests/drafting/test_scaffold_graph.py` unless named).

| Limit | Default | Test |
|---|---|---|
| draft attempts | 3 | `test_the_draft_attempts_are_a_bound` |
| contract repairs | 3 | `test_the_contract_repairs_are_a_bound` |
| smoke repairs | 3 | `test_the_smoke_repairs_are_a_bound_and_the_chain_rolls_back` |
| smoke infrastructure attempts | 3 | `test_the_smoke_infrastructure_attempts_are_a_bound` |
| alignment repairs | 10 | `test_the_alignment_repairs_are_a_bound_and_the_draft_rolls_back` |
| alignment judge calls | 2 * (alignment + qualification repairs + 1) | `test_the_judge_call_budget_is_twice_the_repairs_plus_one` |
| report retries | 1 | `test_a_second_malformed_report_ends_drafting` |
| qualification repairs | 4 (cap 8) | `test_the_qualification_repairs_are_a_bound`, `test_without_repairs_a_qualification_failure_is_final` |
| qualification infrastructure attempts | 3 | `test_qualification_infrastructure_attempts_are_a_bound` |
| gate checks per repair | 12 | `test_the_in_session_gate_checks_are_a_bound` |
| smoke, production-entry and qualification timeouts | 1800 s, 300 s, 57600 s | `tests/drafting/test_gate.py` (passed to the evaluator on each request) |
| legacy repair pool | 4 | `test_the_legacy_workflow_shares_one_repair_pool` |
| resource bytes | 16 GiB | `tests/drafting/test_gate.py`, `tests/drafting/test_scaffold_files.py` |
| byte bounds of the qualification repair prompt (16 KiB, 96 KiB, 16 KiB, 16 KiB, 16 KiB) | | `tests/drafting/test_qualification.py` |
| histories (16 alignment reports, 8 rejected repairs, 64 KiB prompt, 64 KiB diff) | | `tests/drafting/test_qualification.py`, `test_scaffold_files.py` |

## Adaptations of the mechanics

1. **The gate is a protocol with a local default.** Upstream's gate runs the checks in containers,
   behind an authenticated socket service that the repair agents call with a client program. The
   package's `LocalTaskGate` runs the same decisions over a `Workspace` and an `Evaluator`
   (`preflight`, `contract`, `runtime`, `search-smoke`, `final-smoke`, `production-entry`,
   `acceptance-smoke` and `qualification`); the in-session client is the `task_gate` tool with the
   same per-repair request budget. Not ported: the Docker runtime preparation, the install cache,
   the broker and the production-entry canary containers, the socket service, GPU leases and
   quarantine and the GPU preflight, the host-storage failure categories and the sealed-final
   topology smoke. A caller whose harness differs substitutes its own `TaskGate`.
   * `runtime` is the search smoke (upstream's native topology does the same).
   * An evaluator that prepares its own runtime reports an authored dependency or baseline failure
     with `failure_category` `runtime_dependencies` or `runtime_baseline_validation`; a
     `runtime_dependencies` failure is the one that can authorise removing an unused pin.
   * `production-entry` runs both evaluators with the evaluation plane `production_entry` and
     `AGENTIC_TASK_PRODUCTION_ENTRY=1` and expects the evaluator to stop after the first
     completed declared reset. It needs an isolated evaluator, as upstream's needs containers: for
     any other evaluator it is an infrastructure failure
     (`production_entry_requires_container`) and `acceptance-smoke` leaves it out.
   * A smoke or qualification evaluation that is `interrupted` is an infrastructure failure that
     is retried like a bootstrap failure; upstream retries only `runtime_bootstrap` and
     `gpu_allocation_busy`.
2. **Qualification always runs.** Upstream skips the production qualification in its native test
   topology and runs it only in containers. The graph always qualifies; a caller that does not want
   the production workload supplies a gate whose `qualification` check passes.
3. **No sleeps and no overall deadline.** Upstream sleeps 1 s and 2 s between smoke infrastructure
   retries and measures `QUALIFICATION_TIMEOUT_SECONDS` as the deadline of the whole drafting run,
   choosing shorter targeted validations when little time is left. The graph retries without
   waiting and passes each check's own timeout to the evaluator; the run's total time is the
   caller's to bound.
4. **The receipt.** Upstream's qualification receipt binds a container runtime (image id, install
   cache key). The package's `QualificationReceipt` binds the scaffold fingerprint, the digest of
   the publication plan, the evaluators, `requirements.txt`, the baseline's result and the
   declared accelerator. Reuse of a qualification and publication check it against the current
   scaffold; a missing or stale receipt blocks acceptance. The publication plan is computed
   before and after the qualification, and a qualification that changed it is the authored
   failure `qualification_modified_scaffold`.
5. **Tree fingerprints hash each file's SHA-256.** Upstream's `task_tree_fingerprint` streams every
   file's bytes into one digest. The workspace protocol has no streaming read, so
   `drafting.scaffold.tree_fingerprint` keeps upstream's traversal (children sorted by name),
   entries and omissions (`.venv`, `.agentic_runtime.json`, `.agentic_repair_prompt.md`) and hashes
   each file's SHA-256 from the workspace listing instead of its bytes. The value differs from
   upstream's for the same scaffold. The `declared_scaffold` fingerprint of the contract's closure
   is upstream's byte layout (`task_scaffold_fingerprint`).
6. **Transactions use `Workspace.snapshot` and `restore`.** A restore replaces the whole staged
   directory, where upstream restores its mutable files (and, for a tree transaction, the whole
   tree). The workspace protocol cannot read a snapshot, so a tree transaction also keeps the
   bounded source files at the snapshot (at most 4 MiB) in state, from which the diff of a rejected
   qualification repair is computed.
7. **Structured judge.** The alignment judge answers with an `AlignmentReport` through structured
   output; the host validates it with upstream's rules and messages and writes it for audit under
   `_task_validation/round_<n>/`. Evidence is checked against the judge's view, as upstream checks
   it against the view. The judge reads the line numbers its prompt names (`nl -ba`) with a
   read-only `nl` tool. A scaffold or view that changes during the review fails drafting, as
   upstream's check does, although the judge's backend cannot write.
8. **Provider failures.** Upstream retries a transient provider failure four times with delays of
   15, 45 and 90 seconds and ends with exit code 75 when they are exhausted. Inside a session
   `ModelRetryMiddleware` retries; the nodes carry the same four-attempt policy; an exhausted
   failure propagates out of the run for the caller to resume. Exit codes 78 and 87 and the agent
   log writer have no counterpart.
9. **Publication.** Upstream's descriptor-safe copy protocol and its journals are not ported; the
   workspace protocol owns safe paths. Publication is idempotent: bytes already in place are left
   alone, a conflicting file or a stale qualification receipt is refused, and the validation marker
   (the alignment report) is written last. The credential scan covers the planned files and the
   resource files up to 16 MiB each (a larger file cannot be read through the protocol).
10. **The staged task reference.** The gates and the evaluators see the staged copy as the task:
    its `task_md_sha256` is the digest of the canonical `task.md`, which the staged copy must
    equal.
11. **Qualification failure evidence.** Upstream's structured runtime evidence of a failed
    qualification (deadline owners, RPC counts, output excerpts) comes from its container runtime;
    the port carries the evaluator's bounded summary as the diagnostic and keeps the allow-list of
    numeric evidence fields for evaluators that supply them in the report's details.

## Upstream behaviour that the package keeps although it looks odd

* `redact_sensitive_text` shows every non-printable character, a newline included, as an escape
  (`\x0a`), so a diagnostic or a bounded report that reaches a repair prompt through it is a single
  line. The package keeps this (the prompt texts that were tuned on such input are unchanged).
* The typed smoke diagnostic withholds the task-authored text of an authored failure and carries
  a fixed sentence; the text is shown only through the in-session gate.
* The first smoke repair of a `runtime_dependencies` failure says that the dependency transaction
  is `inactive` although removal is already permitted by the policy of that repair; the prompt
  explains that case.
* The alignment repair prompt receives the retained alignment history only; the rejected-repair
  history (source diffs) goes to the qualification repairer, never to the judge.

## Upstream defects found and handled

* **The calibration never reached the judge.** Upstream loads the calibration record for the
  judge's attestation through `redact_sensitive_text`, which escapes the record's newlines, and
  then parses it as JSON, which fails, so `qualification_calibration` is never present in the
  attestation although the judge prompt tells the judge to compare it. The package injects the
  parsed record (and a marker when it exceeds 16 KiB, and nothing when it holds a credential).
  Tested in `test_a_qualification_failure_is_repaired_as_a_pending_transaction`.
* **Short environment values hid every diagnostic.** Upstream looks for the value of every
  environment variable whose name contains `AUTH`, `KEY`, `TOKEN` and similar in a text before it
  is shown; a variable holding `1` or `true` matches everywhere. The package ignores values shorter
  than eight characters.

## Not ported

The agent CLI vendor layer (headless runs, authentication staging, the bounded log writer and its
failure channel), container and GPU handling, the bytecode quarantine (generated bytecode is
removed instead), the repair-prompt file in the workspace (the prompt is the session's message; its
reserved path is still refused), the progress events, the cost tracking and the sanitised audit
copy of the validation directory (the report of each round is kept under `_task_validation/`), the
`TASK_RUNTIME_CACHE_HANDOFF` switch, and `task_setup.py` (which runs before investigation, not
before drafting).
