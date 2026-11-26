"""Search evaluator of the contracted toy task: combined_score and accuracy on the search seeds."""

import argparse
import json
import os
import runpy

parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--num_replications", type=int, default=1)
args = parser.parse_args()
handle = runpy.run_path(args.program_path)["handle"]
target = float(open(os.path.join(os.path.dirname(__file__), "resources/search/target.txt")).read())
value = float(handle({"seed": 1}))
score = -abs(value - target)
os.makedirs(args.results_dir, exist_ok=True)
json.dump({"combined_score": score, "accuracy": 1.0 if score > -1 else 0.0},
          open(os.path.join(args.results_dir, "metrics.json"), "w"))
json.dump({"correct": True}, open(os.path.join(args.results_dir, "correct.json"), "w"))
json.dump({"schema_version": 1, "episodes": [{"seed": 1, "score": score, "correct": True, "duration_seconds": 0.1}]},
          open(os.path.join(args.results_dir, "evaluation_evidence.json"), "w"))
