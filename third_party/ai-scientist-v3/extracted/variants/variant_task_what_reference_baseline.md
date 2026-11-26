<!-- Extracted from src/variants_pipeline/materialize.py lines 519-566 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->

## What To Do
`program_0/` contains a **reference implementation** from an external source. Your task is to **adapt** it to integrate with our evaluation harness for a fair, apples-to-apples comparison.

The **control program** is available at `control_program/`. Study it to understand:
- How `evaluate.py` calls and evaluates the program
- The expected function signatures and metric reporting format
- Data loading API and file structure conventions

### Step 1: Hyperparameter Audit

Before making any code changes, compare the hyperparameters of the reference implementation (`program_0/`) with those of the control program (`control_program/`). Write your findings to `hyperparameter_comparison.md` as a table with these columns:

| Parameter | Reference value | Control value | Decision | Reason |
|-----------|----------------|---------------|----------|--------|

For each parameter, the **Decision** should be one of:
- **align**: This is an experimental condition (not part of the algorithm's contribution) that should match the control for a fair comparison
- **preserve**: This is part of the algorithm's design and must be kept as-is

The guiding question is: *is this difference an unfair advantage/disadvantage, or is it part of how the algorithm works?*

### Step 2: Adapt the Implementation

Create `program_1` by adapting the reference implementation. Use the hyperparameter audit to guide your decisions:
1. Keep the **core algorithm** from `program_0/` unchanged — the scientific contribution must be preserved faithfully
2. Adapt the **interface** to match what `evaluate.py` expects (use `control_program/` as reference for function signatures, metric reporting, and file structure)
3. **Align** all parameters you marked as *align* in the audit
4. **Preserve** all parameters you marked as *preserve* — changing these can silently degrade performance and produce misleading comparisons

### Step 3: Performance Sanity Check

After evaluating `program_1`, compare its performance against the expected performance documented in `program_0/` (e.g. README). If you observe a **large degradation**, revisit your audit — one of your *align* decisions may have been wrong. Revert any change that is not strictly required for `evaluate.py` to run, and re-evaluate.

Do NOT evaluate `program_0` first — go straight to creating and evaluating `program_1`.

You may create up to **{{MAX_EVALS}}** program attempts (`program_1` through `program_{{MAX_EVALS}}`). If a program passes all rubrics, stop immediately — do not use remaining attempts. If it fails, create the next `program_<idx>` as a repair, fixing only the rubric failures.
