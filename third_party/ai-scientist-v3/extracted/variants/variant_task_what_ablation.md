<!-- Extracted from src/variants_pipeline/materialize.py lines 497-505 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->

## What To Do
Create `program_1` by cloning `program_0` and applying ONLY the ablation described above. Do NOT evaluate `program_0` first — go straight to creating and evaluating `program_1`.

You may create up to **{{MAX_EVALS}}** program attempts (`program_1` through `program_{{MAX_EVALS}}`). If a program passes all rubrics, stop immediately — do not use remaining attempts. If it fails, create the next `program_<idx>` as a repair, fixing only the rubric failures.
