# Differences from upstream

This package ports The Agentic Scientist (AI-Scientist-v3, Sakana AI) at commit
`df8313889a004f8a44a472e19a833cd547bb828a` to LangGraph. The flow and the prompts of the original are
carried over; this file lists every place where they differ, and why. An **adaptation** is a step
that exists in another form; an **omission** is something that is not ported.

Where this file and the upstream source disagree about what upstream does, the source is the
authority, and the discrepancy is recorded in the note of the area that found it.

## Where each change is recorded

Every change to a prompt, skill or workflow text is declared in a file of
`src/langgraph_agentic_scientist/prompts/changes/` (an exact old string, the new string, and the
reason). `tools/derive_prompts.py --check` re-derives every package text from the unmodified
original under `third_party/ai-scientist-v3/` and fails on any difference, so the files below are
the complete list of what differs. Each area also has a note of the flow's adaptations and of the
discrepancies found while porting.

| Area | Note | Declared prompt changes |
|---|---|---|
| Core: configuration, context, prompts, workspaces, evaluators, agents, shared records | [adaptations/core.md](adaptations/core.md) | [core.toml](../src/langgraph_agentic_scientist/prompts/changes/core.toml) |
| Task design loop | [adaptations/drafting_design.md](adaptations/drafting_design.md) | [drafting_design.toml](../src/langgraph_agentic_scientist/prompts/changes/drafting_design.toml) |
| Task scaffold drafting and gates | [adaptations/drafting_scaffold.md](adaptations/drafting_scaffold.md) | [drafting_scaffold.toml](../src/langgraph_agentic_scientist/prompts/changes/drafting_scaffold.toml) |
| Investigation | [adaptations/investigation.md](adaptations/investigation.md) | [investigation.toml](../src/langgraph_agentic_scientist/prompts/changes/investigation.toml) |
| Ablations (variants) | [adaptations/variants.md](adaptations/variants.md) | [variants.toml](../src/langgraph_agentic_scientist/prompts/changes/variants.toml) |
| Final evaluation and post-hoc analysis | [adaptations/final_and_posthoc.md](adaptations/final_and_posthoc.md) | [posthoc.toml](../src/langgraph_agentic_scientist/prompts/changes/posthoc.toml) |
| Paper: data, references, validation | [adaptations/paper_host.md](adaptations/paper_host.md) | [paper_host.toml](../src/langgraph_agentic_scientist/prompts/changes/paper_host.toml) |
| Paper: the writing loop | [adaptations/paper_loop.md](adaptations/paper_loop.md) | [paper_loop.toml](../src/langgraph_agentic_scientist/prompts/changes/paper_loop.toml) |
| AI review and revision | [adaptations/review_revision.md](adaptations/review_revision.md) | [review.toml](../src/langgraph_agentic_scientist/prompts/changes/review.toml) |

The systematic changes that the declared prompt changes carry, by kind:

- **A1, helper programs that cross the trust boundary become tools.** Upstream tells an agent to
  run modules of its own package (the trusted program clone, the evaluation wrapper, the task gate
  client). The evaluator and the gate must stay out of the candidate's reach, so these are tools
  (`clone_program`, `evaluate_program`, `task_gate`) with the same arguments and the same rules
  (budget counting, refusal before the rubric artefacts exist, no re-run after an interrupted
  call). Self-contained helpers (the memory helper, the rubric placeholder helper, the paper
  validators) are not changed: they are staged unchanged with their skills.
- **A2, an agent has no exit status.** "Exit non-zero and report the exact blocker" becomes "call
  the `report_blocker` tool with the exact blocker and stop".
- **A3, structured answers.** For a role whose whole product is one structured artefact, the
  sentences that tell the agent to write a JSON file become "return it as your structured
  response"; the host writes the file under the name upstream uses. Field lists, examples and rules
  about content stay.
- **A4, staged skill paths.** Skills are staged as sibling directories under `.agents/skills/` of
  the role's workspace instead of a coding-agent CLI's configuration directory.

## Adaptations of the flow

1. **Coding-agent CLIs become Deep Agents agents over a workspace.** Upstream runs every role as an
   external coding-agent CLI in a working directory. Here each such role is a Deep Agents agent with
   the same kind of tools (files, search, shell, sub-tasks), run as a subgraph of the node that
   needs it, so its steps are checkpointed and a resumed run continues the session. One node runs
   at most one session; every retry of a role is a graph loop with a counter in state. Lost:
   comparison across agent CLIs and their own tool sets and context management. Kept: sampling
   several models per idea cycle, through `ModelRegistry.pool`.
2. **Agent-written JSON becomes structured output for structured roles.** Upstream agents write
   every output, even pure JSON answers, as files that the host re-reads and repairs by hand
   (trailing commas, verdict strings, the audit's line parser). A role whose whole product is one
   artefact now answers through LangChain's structured output, and the host writes the file under
   upstream's name. Nothing of the method is lost; the hand repairs have no object any more. The
   structural-repair step of the design loop stays for semantic schema failures that structured
   output does not prevent. Roles that edit or create files keep writing files.
3. **Trust-boundary helpers become tools; self-contained helpers are staged unchanged.** See A1
   above. The budget of the self-evaluated regime is enforced by the tool instead of being counted
   after the session; upstream's post-session check stays as a second line.
4. **State in files becomes typed state, and files are written as views.** Upstream keeps all state
   in files and resumes from them. Here the host's records (outer state, histories, the best
   manifest, counters, budgets, transactions, reports) are typed graph state with reducers,
   checkpointed by the caller's checkpointer, and the host writes the files that prompts name before
   the agent that reads them runs. What agents write themselves (the memory files, conclusions)
   stays theirs, and the host reads it after a session. Lost: the run directory as the record of a
   run; the caller's checkpointer and the workspace together hold it.
5. **The shared memory directory and its lock become the caller's store.** Memory shared across
   parallel workers goes through the `BaseStore` under deterministic keys, with upstream's
   replace-then-push-delta semantics. The shared order is (cycle, worker) instead of the order in
   which workers happened to finish.
6. **Session wall clocks are kept as a timeout around the agent session in its node**, with
   upstream's values and upstream's consequence (a timed-out session's outputs are still checked).
   Shell commands keep upstream's 600 s default and 7200 s maximum; evaluation timeouts are kept on
   every evaluation request. Lost: after a crash the resumed session's clock starts again, as
   upstream's resumed cycle does. An optional bound on model calls per session is offered
   (`config.agents.model_call_limit`); upstream has none. LangGraph's own node timeout
   (`add_node(timeout=TimeoutPolicy(...))` with an `error_handler`) is not used for this, for three
   reasons. The clock's value is a setting of the run (`run_timeout_seconds` and the other
   per-phase timeouts of `ScientistConfig`, recorded in state and read at run time), whereas a
   node's `TimeoutPolicy` is fixed when the graph is compiled. A timed-out session must leave its
   partial result: the helper returns `timed_out=True` with the messages streamed so far, and the
   graph reads from them the evaluations the agent already paid for and its blocker before checking
   the outputs, as upstream does after a timed-out agent CLI; a `NodeTimeoutError` discards the
   node's partial result, and an error handler sees only the state and the error. And the node's
   timeout would bound the whole node (staging, the session and its post-processing), not the
   session. The clock bounds each uninterrupted run of a session; it restarts after a resume, as a
   `TimeoutPolicy` would. A caller who wants a hard bound on a whole step can still put a
   `TimeoutPolicy` on it through a replacement slot.
7. **Provider retries and sleeps become LangChain's and LangGraph's retry mechanisms.** A transient
   failure inside a session is retried by `ModelRetryMiddleware` with upstream's 15, 45 and 90 s
   schedule, which repeats only the failed model call; one that escapes the session is retried by
   the node's `RetryPolicy` (four attempts, the same schedule), which starts the session again, as
   upstream's retry of a failed agent run does. Both layers are kept, as upstream has both (the
   agent CLI retries its API calls, the harness retries the run): a node retry does not continue a
   session from its checkpoint, so without the inner layer every transient error would restart a
   session of hours. `config.agents.model_retry.max_retries=0` leaves only the node's layer. Plain
   model calls (the AI reviewers) use a `RetryPolicy` with six attempts. Method retries that change
   the prompt (repairs, second attempts, contract retries, the reviewers' second attempt) are graph
   loops with counters, so that each paid call is checkpointed on its own.
8. **Sequential becomes parallel where independence is upstream's.** Five reviewers, the ablation
   variants after the control, the post-hoc analyses and the campaign workers fan out with `Send`.
   Variant results are joined, not chained; the control still runs first, so "a failed control
   aborts the rest" holds. The caller bounds concurrency with `max_concurrency`.
9. **Docker, the broker and sealed containers become `Workspace`, `Evaluator`, `FinalEvaluator` and
   `TaskGate`.** The package ships local implementations that run evaluators as subprocesses
   without isolation; candidate isolation, the RPC broker, the broker's reset-transcript
   enforcement and GPU partitioning are the caller's executor's. The drafting prompts still describe
   the broker contract, which is the method's task contract; a caller whose harness differs
   overrides those prompt entries. Lost with the local defaults: the trust boundary itself, which is
   why a contracted task is refused an evaluator that does not isolate candidates
   (`allow_unisolated_contracted=True` lifts this for local experiments and tests).
10. **Final evaluation launch.** Pre-start launch retries, image re-resolution, broker recovery and
    GPU quarantine belong to the `FinalEvaluator`; the package keeps the once-only rule, the freeze,
    the manifest verification and the digests.
11. **Concept figures.** Autoillustrate stays an outside tool behind `ConceptFigureRenderer`
    (`AutoillustrateRenderer` runs it through the workspace when it is installed, with upstream's
    time-based modes, `--workdir` and the pinned `--model`). Its `--prompt` is the strategy's figure
    purpose unchanged, as upstream sends it. Without a renderer the placeholder figure and the
    prompts' documented plain-export fallback apply, as upstream does without keys.
12. **Reference retrieval.** Semantic Scholar first, then the OpenAlex and `scholarly` scripts, behind
    `ReferenceRetriever`; a caller may substitute its own service.
13. **The AI reviewer's model.** Upstream calls one hosted model; the port calls the models of the
    `reviewer` and `area_chair` roles. Lost: comparability of scores with upstream's calibration,
    which upstream did not validate either.
14. **Review consent.** As upstream, selecting the `review` phase is the consent to send the
    manuscript to the reviewer; no question is asked.
15. **Prune confirmation.** Upstream waits for Enter on a terminal, once per campaign. The port asks
    once through `interrupt` or the policy and records the answer in state, so a resume does not ask
    again; `AutoApprovePolicy` behaves as upstream does without a terminal.
16. **Watermark.** Kept, on by default and applied before every compile, with upstream's text.
    `config.paper.watermark` may be turned off by a caller who publishes into its own paper with its
    own disclosure; upstream has no switch.
17. **The paper's parallel-worker merge.** Upstream merges the workers' result directories with
    symlinks for the paper; the paper data are built from all workers' typed records instead.
18. **Placeholders are rendered in one pass.** Upstream's sequential replacement re-expands
    placeholders inside substituted values, which is a defect; the port does not.
19. **The `legacy` drafting workflow** is an option (`config.drafting.workflow`), off by default as
    upstream.
20. **The configuration is recorded with the run.** Upstream persists only the idea-cycle budget as
    its trusted resume policy. The port records the whole method configuration and the prompt
    fingerprint in state and refuses a resume that changes them.
21. **Figure-quality repair without containers.** Upstream attempts it only when containerised,
    because only a container session can be stopped before the snapshot is restored. A port session
    has ended when its node returns, so the repair is attempted whenever `Workspace.snapshot`
    succeeds.
22. **Caller hooks and the sealed-input draw are graph nodes.** Upstream has no hooks; the port's
    run once and are checkpointed.
23. **Read-only roles** run on a read-only backend (and the evidence auditor on a disposable copy
    with an input-hash check) instead of upstream's read-only container copy. Without a container
    the port still guarantees that the role cannot write, which upstream guarantees by refusing
    native execution.
24. **Skills are staged under `.agents/skills/`** instead of a CLI's configuration directory (A4).

### The pipeline

These concern the orchestrator (`build_pipeline_graph`), which replaces upstream's `agsci run`.

25. **One graph instead of subprocesses.** Upstream shells out to one process per phase and
    forwards its configuration as Hydra overrides. The port mounts each phase's compiled graph
    through a wrapper node that passes the phase the keys of its input schema that the pipeline
    knows and returns only the phase's result as a delta. Phases are replaceable, and the pipeline
    keeps no persistence of its own: the caller's checkpointer resumes a run, so upstream's
    resume anchors (`paths.run_dir`, the resume policy, the completion receipts) have no
    counterpart. A run that starts at a later phase takes the earlier results in its input.
26. **The task snapshot** is made once into `task/` of the workspace, honouring
    `.agentic_task_ignore`, refusing absolute and escaping symlinks, and recording a tree digest in
    `.agentic_task_snapshot.json` (upstream's name). A snapshot without that file is an interrupted
    copy and is made again; an existing one is verified against its digest and never refreshed from
    its source. After drafting publishes a scaffold the snapshot is resealed, as upstream reseals
    it. The source is a path inside the workspace; upstream's descriptor-safe copy protocol, ready
    markers and root-mode record are the workspace's business.
27. **Failing fast.** `config.fail_fast` is upstream's `execution.fail_fast` (default true, as
    upstream's): with it a failed phase that cannot be skipped ends the run, and without it the run
    goes on and the failed phase's dependants are skipped. Optional phases degrade either way. The
    terminal host failures (`evaluator_output_contract_invalid`,
    `agent_log_persistence_failed`) and a drafted task that needs an unselected final evaluation
    stop the run whatever the setting says. A run that stops records the phases it never reached as
    skipped with the reason `run stopped: <category>`.
28. **The final-evaluation dependency.** Upstream refuses, before anything runs, a paper phase for
    a task whose contract requires a final evaluation when the final phase is not selected and the
    policy is not `allow_not_qualified`; once drafting has published such a contract the same
    condition stops the run whatever the policy says. Upstream then reads the final evaluation from
    the run directory; the port takes it from the phase, or from the input of a run that starts
    later. Both checks are kept as upstream has them.
29. **Phases that are not selected, or are skipped, are recorded** in the output's `phase_status`
    with a reason (`not selected`, `dependency investigation failed`, `investigation completed
    without a qualifying winner`, `selected ablations did not succeed`, `required final evaluation
    did not qualify`), and each phase result is stored under the pipeline's own key.
30. **Phases drop out on resume by themselves.** Upstream drops task drafting on resume unless the
    phases are explicit. A resumed run already holds its phases and its drafting result in its
    checkpoint, and a new run on an existing workspace skips drafting because the scaffold is
    present.
31. **The harness incumbent is established once per campaign.** Upstream reads the baseline's score
    from the task runtime's validated record and every worker reconciles against it. The port's
    investigation graph takes it, before the first round, from the caller's `incumbent`, from the
    task's qualification receipt (when the seed is the task's `program_0` and the receipt binds the
    task's evaluators as they are now), or from one evaluation under `<run>/baseline`, and passes
    it to every worker (`docs/adaptations/investigation.md`).
32. **The harness-evaluated author's `edit-program` skill states the stage's plain copy.** Upstream
    staged the generic skill, whose trusted-clone command its stage prompt overrides; in the port
    the generic skill names a `clone_program` tool that this role is not given, so the role is
    staged a stage-specific version whose clone step is the plain copy its prompt prescribes
    (`skills/edit-program-harness`, declared in `prompts/changes/core.toml`).

## Omissions

1. Headless CLI, agent CLI installation, authentication staging, billing selection and paid
   fallback, capacity retries and the exit codes 75, 78 and 130: they belong to the CLI vendor
   layer.
2. Docker image build, container options, `.env` mounting and the Docker socket rules: the
   caller's executor.
3. GPU leases, quarantine and ownership journals: bound to Linux, Docker and NVIDIA hosts; a
   caller's executor owns devices. Lost: protection against reusing a GPU whose evaluator may still
   run.
4. Git sidecar checkpoints with secret scanning and size caps: the checkpointer replaces resume.
   Lost: a browsable git history of artefacts and its credential scanner.
5. The cost ledger and token accounting: standard LangChain callbacks carry usage metadata.
6. Event logging, the heartbeat, the terminal UI, the run inspector and AgentLens: tracing is the
   caller's callbacks.
7. `agsci doctor`, `--dry-run` and the Hydra CLI: the package has no CLI; `ScientistConfig`
   validation and the pipeline's configuration errors replace the config checks.
8. The temporary-storage system (`AGENTIC_TEMP_ROOT`, socket roots): a workspace concern.
9. Publication journals, descriptor-safe copy protocols and portable run storage: resume is the
   checkpointer's, and the workspace protocol owns safe paths.
10. Agent-log persistence failure (exit 87) as its own category: it maps to "agent run could not be
    recorded" from the agent runner.
11. The idea placeholder helper, the Pi model list, the in-code fallback prompts, and a stale query
    builder that no upstream code calls.
12. The GRPO example.
13. Dormant upstream paths: the three-judge design quorum with its adjudicator prompt, which the
    upstream loop does not wire (it runs one judge), and the self-evaluating variant agent prompt,
    which no upstream code loads. Both prompts are kept unchanged under `third_party/`; there is no
    graph branch, switch or test for them, because there is no upstream behaviour to match and
    wiring them would mean inventing how their placeholders are filled and their results merged.
14. Upstream's exit 130 (interrupted by a person at the terminal) and the session quiescing around
    it: a run is resumed from its checkpoint.
15. The orchestrator's checkpoint-topology, GPU-allocation and image-pinning preflights, the
    revision-publication recovery, and the dry-run and inspection paths of `agsci run`: they
    concern directories, devices and images that a caller's workspace and executor own.
