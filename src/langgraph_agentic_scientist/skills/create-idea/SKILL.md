---
name: create-idea
description: Create one next-step idea bundle and matching rubric bundle for iterative optimization. Use during idea generation runs (for example via scripts/idea.sh) when `idea_bundle.json` and `target_rubrics.json` must be produced.
---

# What the skill is about
Produce one concrete, testable next experiment idea with machine-readable rubrics.


# Output creation
Return the idea bundle as your structured response. Do not write the JSON files yourself:
the harness writes both required JSON files from the structured response.


# What to do
1. Read task context:
- `{{TASK_MD_PATH}}`
- `{{TASK_GOAL}}`
- `{{TASK_RUBRICS}}`
2. Use cycle and history context:
- `cycle_index={{CYCLE_INDEX}}`
- recent ideas and attempts from the prompt context
- best-known manifest from the prompt context
3. Return one idea bundle as your structured response; the harness writes exactly these files from it:
- `{{IDEA_BUNDLE_FILE}}`
- `{{TARGET_RUBRICS_FILE}}`
4. The structured response must be an object with keys:
- `idea_id` (string)
- `title` (string)
- `hypothesis` (string)
- `implementation_plan` (non-empty array of short strings)
- `target_rubrics` (non-empty array of `{id, description, type}`)
- `notes_for_agent` (string)
5. Rubric rules:
- include at least one `metric` rubric
- include at least one `implementation` rubric
- use stable snake_case rubric ids
- keep rubric descriptions concrete and measurable
6. The harness writes `{{TARGET_RUBRICS_FILE}}` as exactly `idea_bundle.json.target_rubrics`.
7. Apply the active harness rules exactly:

{{HARNESS_RUBRIC_REQUIREMENTS}}

When a rubric includes `verification`, `verification` must be a JSON object, never a prose string.
For harness implementation rubrics, use exactly this shape:

```json
{
  "verification": {
    "type": "source_contains",
    "patterns": ["specific executable snippet"]
  }
}
```

8. Before returning, reread your bundle and confirm every implementation rubric satisfies the active harness rules.
9. Do not modify files outside the required output files.
