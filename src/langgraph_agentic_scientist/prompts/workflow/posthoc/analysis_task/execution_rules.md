
## Execution Rules
Use `task_bundle/program_0` as a read-only artifact-ready base. This post-hoc pipeline is analysis-only: do not retrain, fine-tune, call `evaluate.py`, or create new program variants. Focus on analysis scripts that inspect existing logs, metrics, saved artifacts, and checkpoints when present. Repairs happen in the next `attempt_<n>` directory, not by creating a new analysis workspace.
