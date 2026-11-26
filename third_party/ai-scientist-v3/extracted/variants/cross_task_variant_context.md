<!-- Extracted from src/variants_pipeline/cross_task.py lines 708-718 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->

## Context

`program_0/` contains the **control program already adapted** to the target task (**{{TASK_NAME}}**). It is a working baseline on this task.

`original_variant_program/` contains the **variant program** from the original task. It implements the algorithmic changes described below.

Your job is to **port the variant's changes** onto `program_0/` so that the variant runs on the new task.

- `source_task.md` describes the **original task** both programs were built for.
- `original_task.md` describes the **target task**.
- `evaluate.py` is the evaluator for the **target task**.
