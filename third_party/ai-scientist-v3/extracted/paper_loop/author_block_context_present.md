<!-- Extracted from src/investigation/paper_prompt_context.py lines 169-176 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->
A canonical author-block file is provided for this run.
Read `{{AUTHOR_PATH}}` before drafting or repairing `paper.tex`.
Treat its contents as a hard contract for the `\author{...}` block.
Keep `\input{paper_authors.tex}` inside `\author{...}` when possible; otherwise copy the exact file contents without renaming, reordering, anonymizing, or replacing authors or affiliations.