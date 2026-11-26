# Paper writing loop: adaptations and the two transition tables

This note covers `src/langgraph_agentic_scientist/paper/` except the host side (paper data,
references, validation, watermark, venues), which has its own note. It records how upstream's
`src/scripts/paper.sh` (a fresh paper) and `src/scripts/paper_edit.sh` (an edit of an existing
paper, and its `--external-review` revision mode) became `build_paper_graph`, every change made to
the prompts of the loop, and every place where the port departs from upstream and why. The
declared text changes are in `src/langgraph_agentic_scientist/prompts/changes/paper_loop.toml`.

## The graph

Preparation: `record_run_config` -> `prepare` (modes, output directory, aliases) -> `paper_data`
(slot `paper.data`; a prebuilt `paper_data` is kept) -> `check_final_outcome` -> `seed_edit`
(edit and revision only) -> `stage_inputs` (`VENUE.md`, `paper_authors.tex`, `contributions.md`,
`global_memory_summary.md`, writer skills) -> `references` (slot `paper.references`) ->
`strategy` (two-stage only; repair rounds) -> `resolve_family` -> `rank_posthoc` (when analyses
exist; repair rounds) -> `augment_paper_data` -> `init_concept_figure` -> the loop.

The loop's decisions are pure functions of `paper/graph.py` (`decide_select_stage`,
`decide_after_write`, `decide_postdraft`, `decide_after_validate`, `decide_after_audit`,
`decide_review_check`, `decide_after_review`, `decide_figure_quality`, `decide_loop`); a node
applies the decision's effects (restore or discard the figure-quality snapshot, save a revision
PDF, authorise a concept-figure repair) and routes by its `next`. Every counter and flag
(`repair_round`, `review_round`, `figure_quality`, `previous_failure_fingerprint`, `reverted`,
`stage`) is state, so a resume re-enters the loop at the row that was running.

Replaceable slots and what they return: `paper.data` (`paper_data`, or `failure`),
`paper.references` (`references`), `paper.postdraft_enrichment` (`postdraft_status`,
`postdraft_detail`), `paper.figures` (`concept`), `paper.validation` (`validation`,
`validation_passed`, `validation_fingerprint`, `validation_soft_warnings`), `paper.audit`
(`audit`), `paper.review_gate` (`in_loop_review`). A slot is a compiled graph sharing the keys of
`PaperState`, or an async callable `(state, runtime) -> update`.

## Fresh paper: transition table (`paper.sh` lines 3825-4191)

`R` is `repair_round`, `Rmax` is `paper.max_repair_rounds` (10), `V`, `A`, `Q` are the outcomes of
validation, the evidence audit and the in-loop review. Each row has its own test in
`tests/paper/test_loop_transitions.py` (`test_F<row>_...`); the runs of
`tests/paper/test_loop_fresh.py` cover the paths end to end.

| # | Node | Condition | Effect and next node | Upstream |
|---|---|---|---|---|
| F01 | `select_stage` | figure-quality repair pending | `stage=figure_quality_repair`, clear pending -> `write` (figure-quality repairer) | 3839-3848 |
| F02 | | `R == 0` | `stage=draft` -> `write` (writer prompt) | 3849-3855 |
| F03 | | otherwise | `stage=manuscript_repair` -> `write` (same writer prompt; the repair context is in the feedback files) | 3856-3864 |
| F04 | `after_write` | session succeeded and outputs present | -> F08 | 3874-3897 |
| F05 | | failed, figure-quality stage | restore the snapshot, `reverted` -> F08 | 3907-3924 |
| F06 | | failed, `R >= Rmax` | incomplete "exhausted repair rounds after R attempts" | 3927-3929 |
| F07 | | failed otherwise | stage-execution feedback in `validation_feedback.*`; `R += 1` -> F01 | 3930-3932, 1722-1753 |
| F08 | `postdraft_gate` | `R == 0`, marker absent | enrichment (gaps -> plan -> retrieval -> merge -> citation weave); its failure ends the phase failed | 3936-3953 |
| F09 | `figures` | not `reverted` | concept figure prepared with `refresh`; failure recorded, not fatal | 3955-3972 |
| F10 | `validate` | | toolchain gates -> `validation_report.json`, `validation_feedback.*`, round archived | 3978-3983 |
| F11 | `after_validate` | figure-quality stage, `V` failed, not reverted | restore the snapshot, `reverted`, validate again | 3995-4010 |
| F12 | | `V` failed / passed | failure fingerprint compared with the previous one (`failure_repeated`) / cleared | 4020-4038 |
| F13 | `audit` | | read-only audit of a disposable copy; outcome pass, findings, operational failure or contract exhausted | 4040-4045, 3510-3653 |
| F14 | `after_audit` | figure-quality stage, not reverted, `A` not pass | restore the snapshot, `reverted`, treat `A` as pass | 4067-4078 |
| F15 | | figure-quality stage, not reverted, `A` pass | discard the snapshot | 4079-4084 |
| F16 | | `A` findings, `R >= Rmax` | incomplete "paper evidence audit reported revision findings after exhausting repair rounds" | 4085-4089 |
| F17 | | `A` findings | authorise the concept-figure repair, save the audit revision PDF, `R += 1` -> F01 | 4090-4094 |
| F18 | | `A` operational failure | incomplete "paper evidence audit failed operationally" | 4096-4099 |
| F19 | | `A` contract exhausted | not gating -> F20 | 4100-4102 |
| F20 | `review_check`, `review_gate`, `after_review` | `review_round < paper_review_rounds` | review; stage failure -> incomplete; `review_round += 1`; `R >= Rmax` -> incomplete; else save the review revision PDF, `R += 1` -> F01 | 4104-4131 |
| F21 | `figure_quality_gate` | `V` passed, actionable warnings, not attempted, `R < Rmax` | attempted; a snapshot that succeeds makes the repair pending, `R += 1` -> F01; a snapshot that fails reports the warning | 4133-4153 |
| F22 | | `V` passed, warnings, otherwise | warning reported once | 4154-4159 |
| F23 | `loop_decision` | `V` passed | -> `finalize` (complete) | 4162-4164 |
| F24 | | `V` failed, `failure_repeated` | incomplete "paper validation errors and inputs were unchanged after repair" | 4167-4170 |
| F25 | | `V` failed, `R < Rmax` | `R += 1` -> F01 | 4171-4175 |
| F26 | | `V` failed, `R >= Rmax` | incomplete "paper validation failed" (and the audit's reason when it had findings or failed) | 4176-4183 |

An incomplete paper is not finalised: upstream exits with status 2 before
`finalize_paper_artifacts` (4186-4191); the port returns `PaperResult(status="incomplete")` with
the reasons joined by `"; "`, as `mark_incomplete_paper_pipeline` joins them, and writes the
validation feedback when it is missing.

## Edit and revision: transition table (`paper_edit.sh` lines 4252-4552)

Every exit of `paper_edit.sh` is a failure (there is no incomplete state). Tests:
`test_E<row>_...` in `tests/paper/test_loop_transitions.py` and the runs of
`tests/paper/test_edit_mode.py`.

| # | Node | Condition | Effect and next node | Upstream |
|---|---|---|---|---|
| E01 | `select_stage` | figure-quality repair pending | figure-quality stage | 4263-4272 |
| E02 | | `R == 0` | draft with the edit writer prompt (`paper_edit.md`, `paper_edit_standalone.md`, or `paper_revision.md` in revision mode) | 4273-4279 |
| E03 | | otherwise | manuscript repair with the same prompt | 4280-4288 |
| E04 | `select_stage`, `after_write` | revision mode | restore the preserved `references.bib` and `references_provenance.json` before the round and after the write | 4259, 4341, 497-545 |
| E05 | `after_write` | session succeeded and `paper.tex`, `claims_grounding.json`, `method_scope_audit.json`, `method_scope_audit.md` present | -> E09 | 4290-4309 |
| E06 | | failed, figure-quality stage | restore the snapshot, `reverted` -> E09 | 4312-4329 |
| E07 | | failed, `R >= Rmax` | failed "exhausted repair rounds after R attempts" | 4332-4334 |
| E08 | | failed otherwise | `R += 1` -> E01 | 4335-4337 |
| E09 | `postdraft_gate` | `R == 0`, not `skip_postdraft_enrichment`, marker absent | enrichment; its failure fails the phase | 4343-4356 |
| E10 | `figures` | not `reverted` | concept figure with the run's `figure_refresh_mode` (`preserve` keeps an unchanged concept state) | 4358-4371, concept_figure.sh 1561-1576 |
| E11 | `validate` | | toolchain gates plus the method scope audit (`validate_method_scope_audit`) as hard errors and the auxiliary check | 4373-4378, 3656-3728 |
| E12 | `after_validate` | figure-quality stage, `V` failed, not reverted | restore, `reverted`, validate again | 4390-4404 |
| E13 | | figure-quality stage, `V` passed, not reverted | discard the snapshot | 4405-4410 |
| E14 | | `V` failed / passed | failure fingerprint / cleared | 4415-4432 |
| E15 | `review_check`, `review_gate`, `after_review` | `review_round < paper_review_rounds` | review; stage failure fails; `review_round += 1`; `R >= Rmax` fails; else save the review revision PDF, `R += 1` -> E01 | 4434-4456 |
| E16 | `review_check` | no review this round, `V` passed | required evidence audit; `V` failed skips it | 4458-4460 |
| E17 | `after_audit` | `A` findings, `R >= Rmax` | failed "evidence audit findings remain after R repair rounds" | 4467-4471 |
| E18 | | `A` findings | `R += 1` -> E01 | 4472-4473 |
| E19 | | `A` operational failure or contract exhausted | failed "required evidence audit failed operationally" | 4475-4478 |
| E20 | `figure_quality_gate` | `V` passed, actionable warnings, not attempted, `R < Rmax` | snapshot -> pending, `R += 1` -> E01; a failed snapshot reports the warning | 4481-4501 |
| E21 | | warnings otherwise | reported once | 4502-4507 |
| E22 | `loop_decision` | `V` passed | -> `finalize` | 4510-4512 |
| E23 | | `V` failed, repeated | failed "validation errors and inputs were unchanged after repair" | 4513-4516 |
| E24 | | `V` failed, `R >= Rmax` | failed "exhausted repair rounds after R attempts" | 4517-4520 |
| E25 | | otherwise | `R += 1` -> E01 | 4521 |

Revision mode is edit mode with the options of `src/pipeline/revision.py` (`build_revision_command`,
lines 175-187): `reference_mode="preserve"` (`refresh` is refused, as `--external-review` refuses
it), `figure_refresh_mode="auto"`, `skip_postdraft_enrichment=True`, no two-stage strategy, the
revision writer (`paper_revision.md`), and the bounded, untrusted `revision_feedback.json`/`.md`
staged from the external review with its 16 kept fields (`stage_external_review_feedback`).

## Text changes (see `paper_loop.toml`)

- Writer prompts (`paper`, `paper_standalone`, `paper_detailed_grouped`, `paper_edit`,
  `paper_edit_standalone`): A4 only, where they name
  `src/agents/skills/validate-paper/templates/figure_manifest.schema.json` (the staged copy is
  under `.agents/skills/`). A8 changes no text: `{{PAPER_OPENAI_API_KEY_CONTEXT}}` and
  `{{PAPER_GEMINI_API_KEY_CONTEXT}}` are filled from `ScientistContext.figure_renderer`
  (`AutoillustrateRenderer` gives upstream's sentences from `paper.sh` lines 536-551; with no
  renderer, "No concept-figure renderer is configured for this run."). The task-specific line of
  `paper_edit.md` (and `paper_edit_standalone.md`) about classification and RL/PPO is kept as
  upstream wrote it; a caller whose task differs overrides the prompt entry.
- `paper_strategy.md`, `paper_posthoc_ranking.md`, `paper_reference_gaps.md`,
  `reference_query_plan_postdraft.md`, `review_json.md`, skill `review-paper`: A3 (structured
  answers; the harness writes the files under upstream's names). Beyond the lines the design
  listed, line 70 of `review_json.md` ("Write valid JSON only, with this exact top-level shape")
  and lines 7 and 24 of the `review-paper` skill ("write a strict JSON review artifact", "Write
  only `paper_review.json`") name the file the agent writes and are changed the same way.
- `evidence_audit.md`: A5. The output contract (lines 64-100) describes the structured
  `AuditPayload` in the same words; both examples are kept as JSON objects; the count line and the
  separator rules are dropped.
- `paper_figure_quality_repair.md`: verbatim.
- Workflow texts (`prompts/workflow/paper/*`), one file per text: the context paragraphs of
  `render_prompt` (`paper.sh` 1366-1424, `paper_edit.sh` 1928-1943), the context strings of
  `paper_prompt_context.py`, the key contexts of `paper.sh` 536-551, the ranking repair suffix
  (`paper.sh` 3077-3078; A3), the editing contract and linkage lines of `paper_edit_context.md`,
  the instruction lines of the audit and review packets, the header of
  `global_memory_summary.md`. Every one is extracted by `tools/derive_prompts.py`: shell strings, Python heredocs of the
  scripts and lists of lines included.

## Adaptations of the flow

1. **Structured roles.** The strategist, the post-hoc ranker, the draft-gap analyst, the
   post-draft planner, the evidence auditor and the in-loop reviewer answer through structured
   output; the host writes `paper_strategy.md`/`.json`, `posthoc_ranking.json`,
   `draft_reference_gaps.md`/`.json`, `postdraft_reference_query_plan.md`/`.json`,
   `paper_evidence_audit.json` and `paper_review.json`, then runs upstream's validators on the
   files. Upstream's "contract" failure (an invalid artefact) is a session that ends without a
   valid structured answer; a session that fails otherwise is upstream's non-zero exit. The
   strategist keeps a writable workspace, because its memory protocol appends to the memory files.
2. **The evidence auditor** runs on a disposable copy beside the paper directory (the agents'
   skill directory excluded, as upstream's copy excludes agent namespaces), through a read-only
   backend, and the input hashes of the copy are compared with the paper directory's afterwards
   (upstream `finalize_audit`). The hash covers upstream's audit inputs (watermark block and
   volatile verifier fields stripped), every evidence source by upstream's extension list, and
   the graphics files under `figures/`; upstream additionally hashes TeX includes, bibliography
   styles and figure paths resolved through `\graphicspath`, which the host validator owns.
3. **Figure-quality repair without containers** (design adaptation 21): attempted whenever
   `Workspace.snapshot` succeeds; the snapshot is restored with `Workspace.restore`.
4. **Concept figures**: the renderer is `ScientistContext.figure_renderer` (default none). The
   rendered figure is kept beside the paper directory (`.concept_figure/`) and re-asserted after a
   repair unless an audit finding about it authorised the repair. Autoillustrate's mode is chosen
   as upstream chooses it when a wall-clock budget is known; the port has no workflow deadline,
   so the preferred `full` mode is used. The `--prompt` text is the request's instructions
   unchanged, as upstream sends exactly the strategy's figure `purpose` (or its fallback
   sentence); the example sentence of the `generate-concept-figure` skill is documentation, not a
   prefix, and is not sent. `AutoillustrateRenderer` passes upstream's `--workdir
   ./.autoillustrate/<figure id>` and upstream's pinned `--model` for the `claude` and `gemini`
   agents; `--profile` and `--effort` are constructor options (upstream reads them from the
   agent's environment). Not carried: upstream's environment allow-list for the renderer process,
   its `--help` probe for `--prompt-file` (the renderer always uses `--prompt`), and its
   timeout wrapper, which `Workspace.run` replaces.
5. **Aliases are copies.** `task/`, `results/`, `aggregated/`, `extra_tasks/` are copied into the
   paper directory; of the global memory directory only `memory.md`, `memory.jsonl`,
   `conclusions.md` and `conclusions_history.jsonl` are copied. Upstream symlinks them, so the
   writers' memory appends reached the campaign's files; the port appends the fallback `paper`
   memory entry to both copies at finalisation, and a caller that needs the writers' own appends
   reads them from the paper directory.
6. **Output directory**: `<run root>/paper/final_<run_id>` (upstream `final_<timestamp>`), or the
   input `paper_dir` (or `output_dir`, upstream's `--output-dir`); `latest_manifest.json` is
   written into `manifest_dir` (upstream's `--manifest-dir`), by default the output directory's
   parent, as upstream writes it into `<results>/paper/`. The pipeline passes `paper` for both,
   and the revision its attempt directory for both, as upstream's pipeline and revision commands
   do. Manifest paths are workspace paths.
7. **Writer skills** are staged when the loop prepares the paper directory (and again when the
   strategy changes the writer family), and `watermark_staged_templates` is applied to the staged
   templates then, unless `config.paper.watermark` is off. The writer, strategist and ranker sessions are given those skills without restaging
   them (the in-loop review gate, which can run alone, stages its own `review-paper`), so the templates a session
   sees carry the watermark block, as upstream's do. Host validation still inserts the block
   before every compile, which is upstream's authority.
8. **Post-draft retrieval** searches the plan's queries with the context's retriever and its
   fallbacks, writes `postdraft_references.bib` and the provenance, and merges them with the
   host side's `merge_reference_artifacts`; upstream reran its retrieval script on the plan.
9. **Fresh mode fills `{{PAPER_EDIT_STRATEGY_CONTEXT}}`, `{{PAPER_EDIT_CONTEXT}}` and the scope
   audit paths** with `paper_edit.sh`'s fresh-paper sentences. `paper.sh` leaves them as literal
   placeholders in `paper_strategy.md`, an upstream defect that would show the model raw
   placeholders.
10. **A named main file of the edited paper** (`EditOptions.main_file`). Upstream seeds the
    edited paper's `paper.tex` and a fixed list of artefacts, and snapshots a fixed list of files
    and directories into `previous_paper/`; a paper whose main file has another name, or whose
    text lives in files it inputs elsewhere, is seeded without its manuscript. With `main_file`
    set, `seed_edit` requires that file, writes it as the workspace's `paper.tex`, copies the
    local files it inputs recursively (`\input`, `\include`, local `\usepackage` and
    `\documentclass` files, found by the validator's own dependency walk, which refuses dynamic
    imports, paths outside the paper and symbolic links) to their relative paths unless the
    workspace has them, and adds the main file and its inputs to the snapshot under their own
    names; `paper_edit_context.json` names it as `existing_paper_main_file`. Everything after the
    seeding (the prompts, the skills, the validators, the evidence verifier, the finalisation) is
    upstream's and works on `./paper.tex`, so the finished manuscript is the output directory's
    `paper.tex`. Without `main_file` the seeding is upstream's.
11. **`--venue-file`** is `EditOptions.venue_file`, in every mode: a workspace path is rendered
    into `VENUE.md` with the replacements of `stage_venue_file`, and `""` removes `VENUE.md`, as
    upstream's `stage_optional_rendered_doc` does without a source, so the writer gets the
    venue-absent context. Unset, the format's brief is staged when no `VENUE.md` is present (a
    seeded one is kept). The page budget and the venue checks of the validation follow the venue
    whose LaTeX markers the manuscript carries, as upstream's do: a manuscript without the markers
    of a declared venue has no page budget, and a toolchain given a catalog without that venue
    (`WorkspacePaperToolchain(venues=...)`) applies none.
12. **Final-outcome check before each audit**: `prepare_readonly_paper_audit_workspace` re-reads
    the final evaluation from the run; the port compares `paper_data.json`'s `final_evaluation` and
    `final.*`/`study.final.*` grounding with the paper data the phase holds.

## Omissions

- Exit status 130 and the quiescing of container sessions around it: a run resumes from its
  checkpoint.
- `--resume-existing` fast finalisation (`finalize_resumed_paper_if_valid`): resume is the
  checkpointer's.
- Progress gates, heartbeats, event bridge, cost ledgers and agent logs (`paper_progress.sh`,
  `paper_event_bridge.py`): standard callbacks trace the run; the loop records a few `Event`s.
- `paper_diagnostics` archives of the audit rounds (the validation rounds are archived).
- `prepare_results_dir_for_paper` (merging parallel worker layouts): the paper data are built from
  typed records (design adaptation 17).

## Discrepancies between the design and the source

- The design says a strategy with `status: fail` is fatal. In `paper.sh` the validator raises
  `paper strategy failed: <reason>` as a hard error (2688-2692) and the loop repairs it until the
  rounds are exhausted (2951-2958). The port follows the source.
- The design's table sends an incomplete paper to `finalize`; upstream does not finalise it.
- Row 20 of the design: a review stage failure is incomplete in `paper.sh` but ends
  `paper_edit.sh` (4445-4447).
- In edit mode the audit runs only after validation passes and when no review round remains,
  an exhausted audit contract is fatal (4475-4478), and a figure-quality repair that passed
  validation is not rolled back by the audit (its snapshot is discarded at E13).
