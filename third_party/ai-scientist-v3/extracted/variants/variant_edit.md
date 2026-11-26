<!-- Extracted from src/variants_pipeline/execute.py lines 643-656 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->
You are authoring one scientific comparison candidate.
Use only the edit-program and memory-log skills. Do not evaluate, verify, run Python,
run tests, install packages, or create evaluator artifacts; the host harness owns those steps.

Variant specification:
{{VARIANT_SPEC_JSON}}

Create exactly ./program_{{PROGRAM_INDEX}}/ by cloning ./program_{{PREVIOUS_INDEX}}/.
Implement only the declared variant change. Preserve the CSV analysis infrastructure.
This is edit attempt {{EDIT_ATTEMPT}}. Previous rubric evidence, if any:
{{PREVIOUS_STATUS}}

Stop immediately after the candidate files are complete.
