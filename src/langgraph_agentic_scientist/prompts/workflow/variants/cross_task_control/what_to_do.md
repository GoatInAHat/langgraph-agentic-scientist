
## What To Do

1. Read `original_task.md` to understand the target task requirements.
2. Read `source_task.md` to understand what `program_0` was designed for.
3. Study `evaluate.py` to understand how the target task evaluates programs.
4. Create `program_1` by cloning `program_0` and adapting it:
   - Adapt data loading, input/output formats, and hyperparameters to match the target task
   - Ensure compatibility with the target task's `evaluate.py`
5. Evaluate `program_1` and verify rubrics.
6. If rubrics fail, create the next `program_<idx>` as a repair.

You may create up to **{{MAX_EVALS}}** program attempts (`program_1` through `program_{{MAX_EVALS}}`). If a program passes all rubrics, stop immediately.
