---
name: evaluate-program
description: Evaluate a program for experimentation.
---

# What the skill is about
Run one candidate program with the official evaluator and collect performance and correctness artifacts.

# When to use the skill
- After creating or modifying `{{RESULTS_DIR}}/program_<idx>/src/**` (and/or `program_spec.toml`).
- Before deciding what to try in the next iteration.

# What to do
1. Execute the evaluator in the current environment:

```text
evaluate_program(program_dir="{{RESULTS_DIR}}/program_<idx>")
```

**CRITICAL — execution rules:**
- Always use the `evaluate_program` tool; do not invoke `evaluate.py` directly. Evaluator-owned logs and subdirectories stay outside the candidate tree, while validated summaries are published at the program root.
- `evaluate.py` is a **blocking, self-contained** command. It runs the evaluation, enforces slow-learning guards, and writes all result artifacts before returning. You do NOT need to monitor or poll anything yourself.
- The `evaluate_program` tool call is blocking: it returns when the evaluation has finished. **Never** start `evaluate.py` any other way: not as a background task/process, not through the shell, and not wrapped with the shell `timeout` utility.
- If the shell call ends unexpectedly, first inspect the candidate's `evaluation_manifest.json`. Do not re-run the same program after an interrupted shell call: the call consumed evaluation budget even when no completed manifest exists. Record the failure, write the required memory/rubric artifacts, and stop the attempt.
- **Never** use `sleep` to wait for evaluation results. **Never** read evaluator stdout/stderr logs to check progress. `evaluate.py` handles all of this internally and prints progress to its own stdout.

Notes:
- Entrypoint is resolved automatically from `{{RESULTS_DIR}}/program_<idx>/program_spec.toml` (default: `src/program.py`).
- For a successful evaluation, read `program_<idx>/evaluation_manifest.json` and use its `output_dir` to inspect evaluator-owned logs. For a failed evaluation, use the `evaluation outputs:` path printed by the wrapper. The separate `evaluation runtime:` path contains the evaluator's isolated home and temporary files.

2. Inspect `{{RESULTS_DIR}}/program_<idx>/metrics.json` and maximize `combined_score`.
3. Inspect `{{RESULTS_DIR}}/program_<idx>/correct.json` and require `correct: true`.
4. If needed, debug with logs under the external evaluator output directory reported by the wrapper or manifest.
5. Immediately run `verify-rubrics` before any new edit and write:
- `{{RESULTS_DIR}}/program_<idx>/rubric_status.json`
- `{{RESULTS_DIR}}/rubric_status.json` (same content as latest evaluated program)
6. Never create `program_<idx+1>` until step 5 completed for current `<idx>`.
7. Single-attempt evaluation cap: do not evaluate beyond `program_{{MAX_PROGRAM_EVALS_PER_ATTEMPT}}` (`program_1..program_{{MAX_PROGRAM_EVALS_PER_ATTEMPT}}` only).

# Example usage
```bash
# call the evaluate_program tool: evaluate_program(program_dir="{{RESULTS_DIR}}/program_3")
cat {{RESULTS_DIR}}/program_3/metrics.json
cat {{RESULTS_DIR}}/program_3/correct.json
cat {{RESULTS_DIR}}/program_3/rubric_status.json
```
