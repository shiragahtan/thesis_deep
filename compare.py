"""
compare.py
----------
Run the ablation study: compare smart-only vs dumb-only vs mixed.

This directly answers Q1 from the thesis:
  "Does failure-informed mutation improve sample efficiency?"

Usage:
    python compare.py           # runs all 3 configs, then plots
    python compare.py --quick   # fewer generations (for testing)

Output:
  - results/run_smart_only_<ts>.json
  - results/run_dumb_only_<ts>.json
  - results/run_mixed_<ts>.json
  - results/comparison_plot.png
"""

import argparse
import random
import time
import glob
import os

from config import SEED, SMART_STEP_RATIO, MAX_GENERATIONS
from loop import run
from scheduler import Scheduler
from plot import plot_all


CONFIGS = [
    ("smart_only", dict(strategy="fixed_ratio", smart_ratio=1.0)),
    ("dumb_only",  dict(strategy="fixed_ratio", smart_ratio=0.0)),
    ("mixed",      dict(strategy="fixed_ratio", smart_ratio=SMART_STEP_RATIO)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run fewer generations (faster, for testing)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.quick:
        # Temporarily reduce generations for a quick sanity check
        import config
        config.MAX_GENERATIONS = 5
        config.NUM_PROPOSERS   = 2
        print("Quick mode: 5 generations, 2 proposers\n")

    result_paths = []

    for name, sched_kwargs in CONFIGS:
        print(f"\n{'='*50}")
        print(f"  Running: {name.upper()}")
        print(f"{'='*50}")

        rng       = random.Random(args.seed)
        scheduler = Scheduler(**sched_kwargs, rng=rng)
        result    = run(scheduler=scheduler, seed=args.seed, verbose=True)

        # rename the saved file to include config name
        latest = max(glob.glob("results/run_*.json"), key=os.path.getmtime)
        ts      = int(time.time())
        newname = f"results/run_{name}_{ts}.json"
        os.rename(latest, newname)
        result_paths.append(newname)
        print(f"   Saved as: {newname}")

    # ── Plot comparison ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Generating comparison plot...")
    plot_all(result_paths, save_path="results/comparison_plot.png")

    # ── Print summary table ────────────────────────────────────────────────────
    import json
    print(f"\n{'='*50}")
    print(f"  {'Config':<15} {'Best score':>12} {'Evaluations':>13} {'Target?':>8}")
    print(f"  {'-'*50}")
    for path, (name, _) in zip(result_paths, CONFIGS):
        with open(path) as f:
            r = json.load(f)
        print(f"  {name:<15} {r['best_score']:>11.1f} "
              f"{r['total_evaluations']:>13} "
              f"{'YES' if r['reached_target'] else 'no':>8}")
    print(f"{'='*50}")
    print("\nDone. Open results/comparison_plot.png to see the results.")


if __name__ == "__main__":
    main()
