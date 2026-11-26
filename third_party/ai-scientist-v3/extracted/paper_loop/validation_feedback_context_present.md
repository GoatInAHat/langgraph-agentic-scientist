<!-- Extracted from src/scripts/paper.sh lines 1395-1400 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->
Validation feedback from a previous draft is present.
Read these files before editing:
- `./{{FEEDBACK_JSON_NAME}}`
- `./{{FEEDBACK_MD_NAME}}`
Fix every listed hard error first, keep grounded content stable when possible, and rerun validation after each repair pass until the hard errors are gone.
Treat figure readability findings as soft feedback unless a validator explicitly reports them as hard errors.