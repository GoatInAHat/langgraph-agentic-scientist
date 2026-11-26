<!-- Extracted from src/variants_pipeline/cross_task.py lines 721-737 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->

## What To Do

1. Study `program_0/` to understand how the control was adapted to the new task.
2. Study `original_variant_program/` to understand the variant's algorithmic changes relative to the original control.
3. Create `program_1` by cloning `program_0/` and applying the variant's changes:
   - **Port the algorithmic changes** from `original_variant_program/` onto the adapted control
   - Keep the task-adaptation work from `program_0/` (data loading, I/O formats, etc.) intact
   - Ensure compatibility with `evaluate.py`
4. Evaluate `program_1` and verify rubrics.
5. If rubrics fail, create the next `program_<idx>` as a repair.

You may create up to **{{MAX_EVALS}}** program attempts (`program_1` through `program_{{MAX_EVALS}}`). If a program passes all rubrics, stop immediately.
