# Core: adaptations and discrepancies

This file records how the core package (configuration, context, models, prompts, workspaces,
evaluators, agents, shared records, test support) adapts upstream AI-Scientist-v3 at commit
`df8313889a004f8a44a472e19a833cd547bb828a`, and where the port's documentation and the upstream
source disagree. Every change to a prompt or skill text is declared, with its reason, in
`src/langgraph_agentic_scientist/prompts/changes/core.toml`.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/shared/*.md` (six recipes) | `prompts/fragments/*.md` | none |
| `src/agents/skills/memory-log/SKILL.md` | `skills/memory-log/SKILL.md` | A4: the helper's discovery names `./.agents` |
| `src/agents/skills/memory-log/scripts/append_memory.py` | `skills/memory-log/scripts/append_memory.py` | none |
| `src/agents/skills/edit-program/SKILL.md` | `skills/edit-program/SKILL.md` | A1: the trusted clone is the `clone_program` tool |
| `src/agents/skills/evaluate-program/SKILL.md` | `skills/evaluate-program/SKILL.md` | A1: the evaluation wrapper is the `evaluate_program` tool |
| `src/agents/skills/verify-rubrics/SKILL.md` and its placeholder script | `skills/verify-rubrics/**` | none |
| `src/investigation/reset_sequence.py` (the mismatch diagnostic f-string) | `prompts/workflow/evaluation/reset_sequence_diagnostic.md` | extracted; fields became `{{CONTEXT}}`, `{{EXPECTED}}`, `{{OBSERVED}}`, `{{COUNT}}`, `{{REPLAY_SEED}}` |

The files under `prompts/harness/` are the package's own texts: the agents' system prompt (it
says only that the agent works in the given workspace with the given tools and must not ask
questions) and the descriptions of the tools that replace upstream helper programs.

In the `evaluate-program` skill, the sentences that stay verbatim still speak of "the shell call"
and of the path "printed by the wrapper": the `evaluate_program` tool prints the same
`evaluation outputs:` line the wrapper printed, and the rule those sentences carry (an interrupted
call consumed budget; do not re-run) is enforced by the tool.

## How the method's mechanics are carried

**Agent sessions.** A role runs as a Deep Agents agent invoked from inside a graph node, so that
its steps are checkpointed under the node's namespace and a resumed run continues the session from
its last completed step; a completed tool call is not repeated (tested). One node execution runs
at most one session. The session's wall clock is an `asyncio.timeout` caught in the node: a timed-out
workspace role returns `timed_out=True` with the messages streamed so far, and the graph goes on as
upstream goes on after a timed-out agent CLI. A transient provider failure is first
retried inside the session (`ModelRetryMiddleware`, 15/45/90 s, no jitter), which repeats only the
failed model call and keeps the session's work; one that it could not absorb propagates, so that
the node's retry policy (`agent_node_retry_policy()`, four attempts with the same schedule) runs
the node again, which starts the session again from its prompt, as upstream's retry of a failed
agent run does. The two layers are upstream's too: the agent CLI retries its own API calls and the
harness retries the run. A node retry does not continue the session from its checkpoint (only a
resume after a crash or an interrupt does), so dropping the inner layer would turn every transient
error into a restart of the whole session. A caller that wants one layer sets
`config.agents.model_retry.max_retries=0` or passes its own node policy; any other failure of a session becomes a `Failure` with category
`agent_run_failed`, which the phase's budgets handle as upstream handled a non-zero exit of the
agent.

**Shell budget.** Upstream sets `BASH_DEFAULT_TIMEOUT_MS=600000` and `BASH_MAX_TIMEOUT_MS=7200000`.
The workspace backend uses 600 s as the default command timeout and refuses a requested timeout
above 7200 s. Deep Agents' `FilesystemMiddleware` refuses timeouts above 3600 s by default; the
agent factory replaces it with one whose maximum is the configured 7200 s.

**Deep Agents' own files.** Deep Agents writes offloaded large tool results and conversation
history to its backend. `LocalWorkspace` routes those paths (`/.agent_artifacts/...`) to
`.agent_artifacts/<hash of the role's root>/` at the workspace root, so that a role's directory
holds only what the agent wrote on purpose; upstream's post-checks (for example "only `program_1`
was created") stay true.

**Read-only roles.** `ReadOnlyBackend` refuses writes, edits, deletes and uploads by returning
Deep Agents' error results ("permission denied", which the agent reads) rather than by raising:
an exception raised from a backend method escapes the file tool and would end the session.
Because it is not a sandbox backend, the `execute` tool answers that execution is unavailable.

**Structured roles.** The strategy is explicit per role: `ToolStrategy` for `function_calling`
(default), `ProviderStrategy(strict=True)` for `json_schema`. For plain calls,
`with_structured_output(method=...)` is used; LangChain's base implementation takes no `method`
argument and always uses tool calling, so for such a model `function_calling` is requested without
the argument.

**Skills.** The step-to-skills table is upstream's `SKILLS_BY_STEP` and `PAPER_SKILLS_BY_PROFILE`.
Skills are staged as sibling directories under `<root>/.agents/skills/`; `SKILL.md` is rendered with
the role's placeholder values in one pass, scripts and templates are copied byte for byte, and a
paper writer skill other than `write-paper` receives `write-paper`'s `templates/` tree, as upstream
`stage_paper_templates` does. Agent commands inherit the calling process's environment by default
(`LocalWorkspace(inherit_env=True)`), as upstream's agent CLIs do, so that `python` is found.

**The trusted clone** (`clone_program`) applies upstream's readiness rules and copy rules through
the workspace protocol. The protocol has no operation that creates symlinks, so a symlink inside a
program is reported and not carried (upstream copies it as a symlink).

**The evaluation wrapper** (`evaluate_program`). The n-th call of a session gets the evaluation id
`<prefix>/eval_<n>`, where n counts the session's earlier `evaluate_program` calls in its
checkpointed messages. A call re-executed after a crash therefore asks the evaluator for the same
id, and the evaluator's idempotence returns the recorded (or re-attached) result instead of
evaluating twice (tested). Every call consumes budget, including failed, interrupted and refused
calls, as upstream counts calls, not folders. A completed evaluation publishes `metrics.json`,
`correct.json` and `evaluation_manifest.json` (producer `isolated_evaluator`) at the program root,
as upstream's wrapper does; the manifest also carries the `evaluation_id`, which lets a re-executed
call recognise its own publication.

**Evaluators.** `LocalCommandEvaluator` and `LocalFinalEvaluator` run the task's evaluator scripts
through `Workspace.run` with upstream's sanitized environment (allowlisted names and prefixes,
credential-like names dropped, a private `HOME` and `TMPDIR`). They keep upstream's output rules
(summary checks, the contract-bound shape of `metrics.json` and `correct.json`, the evidence limits)
and, for the final evaluation, upstream's once-only rule, invocation record and manifest schema 3.
They do not carry the container broker: no reset-transcript check (it needs the broker's
transcript), no launch retries, no GPU leases, and results are not made read-only. Their `isolated`
is false.

**Configuration recorded with the run.** `record_run_config` records the method configuration and
the fingerprint of every prompt and skill file; a later entry of `record_run_config` (a resume that
reaches the next phase, or a mounted phase) raises `ConfigurationMismatch` naming the differing
fields or prompt keys. A run that is resumed in the middle of a phase is protected by
`method_config(state)`: nodes read limits from the record, never from the context.

**Serialisation.** `serde_allowlist()` imports every module of the package and returns the
`(module, class)` pair of every `StateModel` subclass and every enum of the package. With
`JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())` the records, the agents' structured
answers (pass their schemas as extra arguments) and Deep Agents' own state survive a strict
checkpoint round trip (tested under `LANGGRAPH_STRICT_MSGPACK=true`). Verified against the
installed `langgraph-checkpoint` 4.3.2 and `langgraph-checkpoint-postgres` 3.2.1:
`AsyncPostgresSaver.from_conn_string(url, serde=...)` accepts a serializer, the synchronous
`PostgresSaver.from_conn_string` does not (pass `serde=` to the `PostgresSaver` constructor), and
`saver.with_allowlist(...)` extends an allowlist only when the saver's serializer is already
restricted; on a permissive serializer it returns the saver unchanged.

## Interfaces the core adds to the design

* `Workspace.stat` (an entry without following a final symlink), `Workspace.host_path` (the path
  as processes started by `run` see it) and the `env` argument of `Workspace.agent_backend`. The
  first is needed by upstream's symlink rules (task contract, clone); the second by evaluators that
  pass paths to a process; the third by the skills that read `ATTEMPT_ID` and `IDEA_ID` from the
  environment.
* `EvaluationRequest.evaluator` (`search` or `final`): the drafting smoke check runs both of the
  task's evaluators.
* `EvaluationResult.metrics` holds JSON values, because upstream's `metrics.json` of tasks without a
  contract may contain non-numeric entries (the circle-packing evaluator writes nested public and
  private metrics); `combined_score` is still checked to be a finite number.
* `FinalEvaluationResult.compute_usage` is optional, and the result carries
  `manifest_schema_version` and `output_dir`, so that schema 1 and 2 manifests can be replayed.
* `DeferredFinalEvaluator`, the sealed counterpart of `DeferredEvaluator`.
* `LocalCommandEvaluator(workspace)` and `LocalFinalEvaluator(workspace)` take the workspace they
  read from at construction; the protocols' methods take only the request.
* The tree digest is upstream's layout; a symlink, which upstream refuses in a digest tree, is
  hashed as `link\0<path>\0<target>`. A tree without symlinks has upstream's digest.

## Discrepancies between the design notes and the upstream source

* **The campaign preset.** The design summarised `configs/preset/paper.yaml` as 3 idea cycles and 5
  ablation replications. Those are the values of `configs/pipeline_config.yaml`; the preset file
  composed on top of it sets 20 idea cycles, one idea-generation attempt, two evaluations per
  attempt, a 3000 s idea timeout, harness-evaluated attempts, one replication and one evaluation
  per variant, the ICLR format and `allow_not_qualified`, and lists all eight phases. The source
  wins: `ScientistConfig.preset("paper")` is the composition, and `ScientistConfig.preset("pipeline")`
  is `pipeline_config.yaml` alone.
* **Stage defaults versus the orchestrator's.** `ScientistConfig()` carries the stage defaults
  (20 cycles, 4 evaluations per attempt, 1 replication, 2 evaluations per variant, 8-15 references),
  as the design's limits table does. Upstream's `agsci run` uses the orchestrator's values instead;
  use `ScientistConfig.preset("pipeline")` to reproduce it.
* **Supported Python.** The package requires Python 3.12 or later; CI runs 3.12 and 3.13.
* **The installed Deep Agents** resolved by the lock file is 0.7.42; its backends and graph
  assembly used here are unchanged from 0.7.41 (the shell backend is byte-identical).

## Rules for phase packages that the core relies on

* Build phase graphs with `StepGraph`; route out of a hooked step with edges, not with
  `Command(goto=...)` from the step; give `Send` targets no hook nodes of their own (make them
  subgraphs whose inner steps carry hooks).
* Call `run_workspace_role` or `run_structured_role` at most once per node; make role retries graph
  loops with counters in state.
* Do nothing with an effect before an `interrupt`, an `ask_person` or an evaluation through a
  deferred evaluator in the same node.
* Read limits with `method_config(state)`; derive every stored record from `StateModel`.
