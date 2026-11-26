<!-- Extracted from src/orchestrator/paper_review.py lines 189-197 of AI-Scientist-v3 df8313889a00 by tests/paper/test_loop_provenance.py; deriving a package file drops this line. -->
Use this packet as the default review entrypoint. Read the raw artifacts only when this packet is insufficient.

## Expected review focus

- unsupported quantitative claims or weak claim-grounding links
- inconsistencies between paper.tex, claims_grounding.json, and validation artifacts
- unclear scientific framing, overclaiming, or missing caveats
- figure or results descriptions that are hard to interpret
- Supplementary Information completeness, relevance, and grounding