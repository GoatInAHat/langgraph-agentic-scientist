<!-- Extracted from src/variants_pipeline/cross_task.py lines 661-667 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->

## Context

`program_0/` contains the **control (best) program** developed for a different task. Your job is to adapt it to work with the current task (**{{TASK_NAME}}**).

- `source_task.md` describes the **original task** the program was built for.
- `original_task.md` describes the **target task** you need to adapt to.
- `evaluate.py` is the evaluator for the **target task**.
