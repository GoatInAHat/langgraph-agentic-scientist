"""Sealed final evaluator of the contracted toy task."""

import argparse
import json
import os
import runpy

parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True)
parser.add_argument("--results_dir", required=True)
args = parser.parse_args()
handle = runpy.run_path(args.program_path)["handle"]
target = float(open(os.path.join(os.path.dirname(__file__), "resources/final/target.txt")).read())
value = float(handle({"seed": 7}))
os.makedirs(args.results_dir, exist_ok=True)
json.dump({"final_score": -abs(value - target)}, open(os.path.join(args.results_dir, "metrics.json"), "w"))
json.dump({"correct": abs(value - target) < 1}, open(os.path.join(args.results_dir, "correct.json"), "w"))
