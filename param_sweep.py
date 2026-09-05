"""
param_sweep.py
--------------
Ablation study over hyperparameters — no LLM required.

Varies:
  - POPULATION_SIZE  (5, 10, 20)
  - NUM_PROPOSERS    (1, 3, 5, 10)
  - TOP_K_PARENTS    (1, 3, 5)

All runs use dumb-only (no LLM) so this is fully reproducible without an API key.

Output:
  - results/sweep_*.json  (one per config)
  - results/sweep_summary.csv
  - Printed table

Usage:
    python param_sweep.py
    python param_sweep.py --quick    # 10 generations each
"""

import argparse
import csv
import glob
import json
import os
import random
import time
import itertools

import config as cfg
from loop import run
from scheduler import Scheduler


# ── Parameter grid ─────────────────────────────────────────────────────────────

PARAM_GRID = {
    "pop_size":   [5, 10, 20],
    "proposers":  [1, 3, 5, 10],
    "top_k":      [1, 3, 5],
}

SEEDS = [42, 1, 7]   # 3 seeds per config for statistical robustness


def run_config(pop_size, proposers, top_k, seed, max_gen):
    """Run one configuration and return the result dict."""
    actual_top_k = min(top_k, pop_size)
    rng       = random.Random(seed)
    scheduler = Scheduler(strategy="fixed_ratio", smart_ratio=0.0, rng=rng)
    result    = run(
        scheduler=scheduler,
        seed=seed,
        verbose=False,
        max_generations=max_gen,
        num_proposers=proposers,
        population_size=pop_size,
        top_k=actual_top_k,
    )

    return {
        "pop_size":          pop_size,
        "proposers":         proposers,
        "top_k":             actual_top_k,
        "seed":              seed,
        "best_score":        result.best_score,
        "total_evaluations": result.total_evaluations,
        "reached_target":    result.reached_target,
        "elapsed_sec":       result.elapsed_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only 10 generations per config (fast testing)")
    args = parser.parse_args()

    if args.quick:
        cfg.MAX_GENERATIONS = 10
        seeds = [42]
        print("Quick mode: 10 generations, 1 seed\n")
    else:
        cfg.MAX_GENERATIONS = 50
        seeds = SEEDS
        print(f"Full sweep: 50 generations, {len(seeds)} seeds\n")

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total = len(combos) * len(seeds)
    print(f"Total configs to run: {len(combos)} × {len(seeds)} seeds = {total}\n")

    all_results = []
    os.makedirs("results", exist_ok=True)

    for idx, (pop_size, proposers, top_k) in enumerate(combos):
        for seed in seeds:
            label = f"pop{pop_size}_prop{proposers}_topk{top_k}_s{seed}"
            print(f"[{idx*len(seeds)+seeds.index(seed)+1:3d}/{total}] {label} ...", end=" ", flush=True)

            t0 = time.perf_counter()
            row = run_config(pop_size, proposers, top_k, seed, cfg.MAX_GENERATIONS)
            elapsed = time.perf_counter() - t0

            print(f"score={row['best_score']:.1f}  evals={row['total_evaluations']}  ({elapsed:.1f}s)")
            all_results.append(row)

            # Save individual result
            path = f"results/sweep_{label}.json"
            with open(path, "w") as f:
                json.dump(row, f, indent=2)

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_path = "results/sweep_summary.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved → {csv_path}")

    # ── Aggregated table ──────────────────────────────────────────────────────
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        key = (r["pop_size"], r["proposers"], r["top_k"])
        agg[key].append(r["best_score"])

    print(f"\n{'pop':>5} {'prop':>6} {'topk':>6} {'seeds':>6} {'mean_score':>12} {'max_score':>11}")
    print("-" * 55)
    rows = []
    for (pop, prop, topk), scores in agg.items():
        mean_s = sum(scores) / len(scores)
        max_s  = max(scores)
        rows.append((mean_s, pop, prop, topk, scores, max_s))

    for mean_s, pop, prop, topk, scores, max_s in sorted(rows, reverse=True):
        print(f"{pop:>5} {prop:>6} {topk:>6} {len(scores):>6} {mean_s:>12.2f} {max_s:>11.2f}")

    print(f"\nDone. Results in results/sweep_*.json and {csv_path}")


if __name__ == "__main__":
    main()
