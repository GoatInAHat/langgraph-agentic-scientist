<!-- Extracted from src/scripts/paper_edit.sh lines 1353-1365 of AI-Scientist-v3 df8313889a00 by tests/paper/test_loop_provenance.py; deriving a package file drops this line. -->
## Editing Contract

- Treat the current `paper.tex` and `claims_grounding.json` as seeded drafts to revise surgically.
- Prior programs, reasoning, and validated contributions remain in scope unless the new artifacts force a grounded correction.
- Prefer preserving valid sections, tables, captions, and references over rewriting them.
- Update contributions and method text so they remain jointly consistent with the seeded paper and the staged program inputs.
- Before editing the abstract, introduction, method, contributions, or cross-domain generalization claims, inspect the staged program inputs directly and audit which manuscript claims are shared versus setting-specific.
- If a mechanism is absent from one staged implementation, scope that claim explicitly instead of presenting it as universal.
- Use `paper_data.json` together with `./aggregated/` and `./extra_tasks/` as the canonical bridge from evaluated results to task families and variant families.
- If staged new-program inputs align with evaluated variant families in the results bundle, treat that as grounded family-level evidence even when exact file-level provenance is unavailable.
- Do not say a staged new-program input has no evaluation results when the results bundle contains matching evaluated task/variant families. Instead distinguish indirect family-level linkage from direct file-level provenance.
- Before finalizing `paper.tex`, write `./method_scope_audit.md` and `./method_scope_audit.json` summarizing the shared method family, setting-specific mechanisms, and any manuscript claims that must be narrowed.
- Keep the manuscript within the venue page budget; compress or localize changes before expanding scope.