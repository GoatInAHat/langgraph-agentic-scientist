<!-- Extracted from src/variants_pipeline/cross_task.py lines 118-130 of AI-Scientist-v3 df8313889a00 by tools/derive_prompts.py; deriving a package file drops this line. -->
## Reference Evaluator (read-only context)

The host-sealed evaluator below illustrates the program-output conventions expected by the pipeline:
- Forwarding `--seed` and `--results_dir` to the program
- Supporting multiple replications (running the program N times with different seeds, writing results to separate directories)
- Appropriate timeout handling
- Producing `metrics.json` and `correct.json` in the results dir

Do not modify evaluate.py. Cross-task support requires the raw extra-task evaluator to already implement the pipeline CLI contract.

```python
{{REFERENCE_EVALUATOR_RAW}}
```
