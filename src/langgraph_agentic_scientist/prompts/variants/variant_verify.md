The host harness has evaluated ./program_{{PROGRAM_INDEX}}/.
Use only verify-rubrics and memory-log. Do not edit code, run evaluate.py, run tests,
install packages, or create another program directory.

Variant ID: {{VARIANT_ID}}
Variant kind: {{VARIANT_KIND}}
Host correctness: {{EVALUATION_CORRECT}}

Inspect ./program_{{PROGRAM_INDEX}}/metrics.json, correct.json, source, and analysis files.
The host has pre-created ./program_{{PROGRAM_INDEX}}/rubric_status.json with the exact
required rubric IDs. Update only each rubric's boolean `pass` and non-empty string
`evidence`, plus optional `next_recommendation`. Do not replace it with a metrics
summary, add/remove rubric IDs, or write ./rubric_status.json; the host owns identity,
metrics, aggregate `overall_pass`, and the workspace mirror.

Required rubric IDs:
{{REQUIRED_RUBRIC_IDS}}

Previous host validation error: {{PREVIOUS_ERROR}}
Then stop.
