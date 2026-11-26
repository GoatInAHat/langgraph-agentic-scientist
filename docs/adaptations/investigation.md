# Investigation: adaptations and discrepancies

This file records how the investigation phase (ideation, attempts, campaign workers, the
investigation graph and the no-winner downstream selection) adapts upstream AI-Scientist-v3 at
commit `df8313889a004f8a44a472e19a833cd547bb828a`, and where the port's design and the upstream
source disagree. Every change to a prompt or skill text is declared, with its reason, in
`src/langgraph_agentic_scientist/prompts/changes/investigation.toml`.

## Texts carried

| Upstream | Package | Changes |
|---|---|---|
| `src/prompts/idea.md` | `prompts/investigation/idea.md` | A3: the two JSON files become the structured response (the "Required Output" block, the sentence that the rubric file equals the bundle's, "Your only outputs", the memory entry made before returning); one change of kind `other`: the rule against a local `./memory.md` in the staging workspace names `{{IDEA_OUTPUT_DIR}}/memory.md` |
| `src/prompts/general.md` | `prompts/investigation/general.md` | A1, one declared change per site: the clone and the evaluation are the tools `clone_program` and `evaluate_program`; the wrapper command block becomes the tool call; the shell-timeout sentence is removed (the call is blocking and owns its timeout); every budget rule that spoke of "commands that run `evaluate.py`" speaks of `evaluate_program` calls |
| `src/prompts/harness_edit.md` | `prompts/investigation/harness_edit.md` | none; byte-identical |
| `src/agents/skills/create-idea/SKILL.md` | `skills/create-idea/SKILL.md` | A3: file writing and the starter-file helper become the structured response; `scripts/create_idea_placeholder.py` is not carried |
| `src/scripts/idea.sh:840-856` | `prompts/workflow/investigation/harness_rubric_requirements_{on,off}.md` | extracted (`expression` of the Python heredoc) |
| `src/investigation/idea_retry_feedback.py:36-38` | `.../idea_validation_feedback.md` | extracted |
| `src/investigation/public_task_contract.py:75-79` | `.../public_search_contract.md` | extracted; `{{INDENTED_PAYLOAD}}` |
| `src/investigation/files.py:39-49` | `.../prune_warning.md` | extracted; the text of the `prune_confirmation` question |
| context blocks of `idea.sh` and `run.sh`, `harness_feedback.py` | `.../context_blocks.toml` | hand extraction: formats, keys and counts |
| evidence and feedback strings of `harness_eval.py`, `harness_feedback.py` | `.../harness_feedback.toml` | hand extraction |

`tools/derive_prompts.py` extracts the texts of the first four rows. The context formats and the
feedback strings live in many f-strings of several functions and heredocs, which the extractor
does not read as one text. They are written by hand into
`third_party/ai-scientist-v3/extracted/investigation/` with a header naming the upstream files and
lines, and derived into the package like every other text.
`tests/investigation/test_provenance.py` checks that every literal fragment of the two TOML files
occurs in the upstream source, which is the machine check the extractor would have provided.

The sentences of `general.md` that the A1 change list of the design did not name but that count
evaluations by the command that runs `evaluate.py` (lines 29, 42, 43, 45, 46, 100, 136) are
changed too: left as they were, they would invite the agent to run `evaluate.py` through the shell,
outside the budget the tool enforces.

## Where agents work, and how memory reaches them

Upstream's agents work in directories of the results tree and reach the campaign's memory files
by relative paths that leave their working directory (`../../memory.md`). Deep Agents' file tools
cannot leave the backend's root (the shell can), so the port gives each role a root that holds
everything its prompt names:

* the **ideator** works in the worker's results root (the design's choice): the campaign memory
  files, the histories, the best manifest, `ideas/<NN_slug>/memory.md` and the conclusions are all
  under it, and `task/task.md` is a copy of the task for it to read. Its staging directory
  (`_idea_generation/cycle_NNNN_attempt_NN/`) is named, with the host-written idea files, by the
  prompt's placeholders. The prompt's rule against writing `./memory.md` in the staging directory
  would otherwise point at the campaign file, hence the one change of kind `other`;
* an **attempt's agent** works in its experiment directory `ideas/<NN_slug>/` (so `./memory.md`
  is the idea's local memory, `program_<idx>` and `./task.md` are where the prompt says, and
  `{{TASK_DIR}}` and `{{RESULTS_DIR}}` are `.` as upstream renders them). The campaign memory
  files it is told to append to are a **view** of the worker's files, staged under
  `_shared_memory/` in that directory; after the session the host merges what the agent appended
  back into the worker's files (`memory_sync.sync_shared_back`), and the post-check reads the
  records from the worker's files. The merge is `<worker file up to the staged size> + <what the
  agent added>`, so it gives the same result when a crash makes it run twice. An agent whose file
  tool rewrote the whole view (Deep Agents' `write_file` overwrites) keeps its lines; lines
  already recorded are not duplicated.

`memory.jsonl` and the conclusions history are not created empty (upstream creates
`memory.jsonl` empty): Deep Agents' `edit_file` cannot edit an empty file, so an agent without a
shell could not append to it, whereas a missing file is created by `write_file` and by the staged
helper alike. A rewrite of a campaign file by the ideator, which works on the file itself, is
upstream's hazard too: the memory skill says "append only", and the host's delta (the lines
beyond the pre-cycle count) is positional, as upstream's.

The harness-evaluated author gets **no package tools**: its prompt forbids the clone and memory
helpers and restricts it to plain filesystem operations, so it clones with a shell command and
writes memory with its file tools or the shell. Upstream staged the generic `edit-program` skill
for this role, whose step 2 tells the agent to clone with the trusted utility, never with `cp`, and
to stop the attempt if the trusted clone fails; the stage prompt says that its own boundaries
override generic skill instructions, so upstream's effective rule was the plain copy. In the port
the generic skill names the `clone_program` tool, which this role does not have, and an agent
that followed the skill would stop the attempt. The role is therefore staged a stage-specific
version of the skill, `skills/edit-program-harness` (staged under the name `edit-program`;
`HARNESS_EDIT_SKILL_SOURCES`), whose step 2 and example state the plain filesystem copy and that
evaluator artifacts are not carried; every other line is upstream's. The stage prompt
(`harness_edit.md`) stays byte-identical and still overrides the skill, as its own sentence says.
The changes are declared in `prompts/changes/core.toml`. For this regime the harness does not stage
`evaluate.py`, `capture.py` or the task's helper modules into the experiment directory
(upstream copies them and tells the agent not to read them): the evaluator owns them. They are
staged, as read-only views, for a self-evaluated attempt only, as upstream does for uncontracted
tasks. The sealed files (`evaluate_final.py`, the contract, the validation record) are never
staged.

## The flow, step by step

* **Incumbent** (`graph.py`, `establish_incumbent`): upstream reads the baseline's score and
  correctness from the task runtime's validated record, produced once, and every worker
  reconciles its best manifest against it (`main_entry.py` with
  `harness_incumbent.reconcile_harness_incumbent`). The port establishes the incumbent once per
  campaign, before the first round, from the first of: the `incumbent` of the input (a caller that
  holds a validated baseline); the task's published qualification receipt, when the seed is the
  task's `program_0`, the receipt records a finite score and a correctness verdict, and its
  `evaluator_tree_sha256` is the digest of the task's evaluators now (the drafting phase measured
  the baseline in qualification); or one evaluation of the baseline under the campaign-level id
  `<run>/baseline`. Every worker of every round whose seed is the task's seed receives it as
  `seed_best` and evaluates nothing, including a worker that found no correct program in an
  earlier pruning round. A baseline that does not complete with a finite score and a correctness
  verdict ends the campaign with `missing_validated_harness_incumbent`.
* **Worker** (`worker.py`): `init_campaign` freezes the seed under `.baseline-seed/`, ensures the
  memory files and takes the incumbent from `seed_best`. Only a worker graph run alone without
  `seed_best` evaluates the baseline itself, under the id `<run>/w<k>/r<round>/baseline`. `confirm_prune` is the first and only effect of its node
  when it asks. `cycle_start` is idempotent: the model choice is a seeded draw (run, worker,
  cycle) persisted in `outer.agent_cycle_assignments`; the prune archives each file once and
  resets after all are archived; the pull replaces files wholesale; the history views are
  rewritten from state. Histories show only the current chapter (`chapter_start_cycle`); the
  result keeps every record.
* **Ideation** (`ideation.py`): `stage`, `ideate`, `validate`, with a loop back to `stage` and the
  validation feedback appended to the next prompt. The staging directory of each generation
  attempt is `cycle_NNNN_attempt_NN` (upstream adds a timestamp; a deterministic name lets a
  resumed node find its own directory). The host writes `idea_bundle.json` and
  `target_rubrics.json` from the validated bundle and, as `idea.sh` does, appends the
  "auto-healed" memory entry when the agent wrote none. Every rule of `idea_generation.py` and
  `idea_bundle.py` applies to the structured answer; upstream's trailing-comma repair has no
  object any more. The diagnostic fed back is escaped and cut at 2048 bytes; upstream's redaction
  of recognisable credentials (the checkpoint sidecar's secret detector) is not ported, because
  the diagnostic is generated from the validator's own messages about the agent's structured
  answer, not from a log.
* **Attempt** (`attempt.py`): harness-evaluated: `prepare`, `author_candidate`, `check_outputs`,
  `evaluate_candidate`, `verify_rubrics`, `record_memory`, `finish`; self-evaluated: `prepare`,
  `search`, `collect_memory`, `finish`. The checks of `harness_eval.py` are kept, and the bounded
  reading of the candidate's sources (128 KiB per file, 4 MiB, 2048 entries, depth 32; symlinks
  refused) runs in `check_outputs`, before the evaluation, as upstream reads the sources before
  evaluating. A session that ran out of its wall clock is checked like any other and then
  evaluated; a self-evaluated session that timed out fails the attempt (upstream exits 124 and
  skips the post-checks). The post-checks of `run.sh:1432-1617` run in `finish` for both regimes.
* **Harness rubric status** (`harness_eval.py`): the rules of `_rubric_status` are a function of
  `HarnessRubricVerifier`, the default `RubricVerifier`; its strings are package data. The exit
  code upstream saw is derived from the evaluation: completed is 0, a timeout is 124, otherwise
  the evaluator's return code or 1.
* **Terminal failures**: `evaluator_output_contract_invalid` stops the campaign. The
  agent-log-persistence failure (exit 87) has no cause in a port (there is no log writer); its
  category is recognised if a replacement slot returns it.
* **Investigation graph** (`graph.py`): `confirm_prune` asks once for the whole campaign and the
  answer goes to every worker. `plan_round` splits the cycle budget with upstream's
  `split_cycle_budget`, plans a `Send` per worker, and requires a store when workers share
  memory. `join_round` stops on a host-owned failure (and, with `parallel_fail_fast`, after a
  round with a failed worker), records each worker's best correct program as its next seed, and
  `prune_round` starts the next round. `finalize` merges the records; the status is `completed`
  when the best program came from an attempt, `completed_no_winner` when the retained baseline is
  still the best (upstream's `_manifest_retains_baseline`), and `failed` when a worker failed.
* **Downstream selection** (`downstream.py`): upstream's rules over `EvaluationRecord`s; the frozen
  copy is made with the trusted clone (so evaluation results and tool directories are not carried,
  as upstream's `_CODE_EXCLUDES`), and `.downstream_selection.json`, written last, binds the
  digests of the source and the frozen tree. A recorded selection is verified and returned on
  resume; a frozen tree that changed raises `StaleDownstreamSelection`.

## Differences from upstream, and from the design

1. **Dynamic cycle leases become a static split.** Without pruning, upstream's parallel campaign
   leases one cycle at a time to whichever worker is free (`_run_single_round_campaign`); it uses
   `split_cycle_budget` only for pruning rounds. A fan-out with `Send` cannot have workers pull
   from a queue without a scheduler of the package's own, so every round splits its budget
   evenly. With equal cycle durations the two coincide; with uneven ones the static split is less
   efficient.
2. **`parallel_fail_fast`** stops after the round, not on the first failure: the workers of a
   round run concurrently.
3. **The store namespace carries the round**: `("agentic_scientist", <run>, "memory",
   "round_<n>")`, so that a pruning round starts with fresh shared memory (upstream resets the
   shared directory per round). Keys are as the design says, with the *global* cycle number, so
   that they are unique across workers and rounds.
4. **Round directories**: with pruning rounds each round has its own results roots
   (`results/round_NNN/worker_NN`), which is how the port keeps an earlier round's context out of
   the next round's workspaces; a single worker prunes inside its loop, as upstream's single
   campaign does.
5. **The incumbent's gate rubric** is written to `target_rubrics.json` and `active_rubrics.json`
   with its structured fields (`metric`, `operator`, `threshold`), but in state it is a
   `RubricSpec`, which has no such fields; the comparison is also in its description
   (`combined_score > <threshold>`), which `rubric_metric_check` reads, so the verdict is the
   same.
6. **Idea records in state** carry the bundle; `ideas_history.jsonl` and `attempts_history.jsonl`
   are views written from state with upstream's field names where the port has the datum
   (`run_exit_code` is 0 or 1; upstream's `output_tail` and agent fields are absent; the model-pool
   index replaces the agent and model names).
7. **`outer_state.json`** is written as a view for the prompt's sake (`idea.md` and `general.md`
   mention it); the state's own copy is `WorkerState.outer`.
8. **Seeds** are always frozen under the results root; upstream skips the freeze when the seed is
   the validated baseline.
9. **Model pool**: the draw for a cycle is uniform and seeded by (run, worker, cycle), so it is
   reproducible; upstream draws once and persists the result, which the port also does.

## Not verified here

* A run against a real model and a real container evaluator; the tests use scripted models and
  fakes.
* Concurrency of several workers against one Postgres store (the in-memory store is used).
* That Deep Agents' `write_file` overwrites an existing file in every backend (it does in the
  local one; the merge of a rewritten view is written for it).
