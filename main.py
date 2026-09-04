"""
main.py
-------
Entry point. Run the evolutionary search with different configurations.

Usage:
    python main.py                        # default run (fixed_ratio scheduler)
    python main.py --strategy diversity   # diversity-aware scheduler
    python main.py --dumb-only            # baseline: dumb steps only
    python main.py --smart-only           # upper bound: smart steps only

Examples:
    # Run the full experiment (smart + dumb, fixed ratio):
    python main.py

    # Compare smart-only vs dumb-only (ablation):
    python main.py --smart-only
    python main.py --dumb-only
"""

import argparse
import random
from config import SEED, SMART_STEP_RATIO
from loop import run
from scheduler import Scheduler


def parse_args():
    p = argparse.ArgumentParser(description="LLM-Guided Evolutionary Search")
    p.add_argument("--strategy",    default="fixed_ratio",
                   choices=["fixed_ratio", "diversity"],
                   help="Step scheduling strategy")
    p.add_argument("--smart-ratio", type=float, default=SMART_STEP_RATIO,
                   help="Fraction of smart steps (used by fixed_ratio)")
    p.add_argument("--smart-only",  action="store_true",
                   help="Use smart step 100%% of the time")
    p.add_argument("--dumb-only",   action="store_true",
                   help="Use dumb step 100%% of the time (baseline)")
    p.add_argument("--seed",        type=int, default=SEED,
                   help="Random seed")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


def main():
    args   = parse_args()
    rng    = random.Random(args.seed)

    # Build scheduler from args
    if args.smart_only:
        scheduler = Scheduler(strategy="fixed_ratio", smart_ratio=1.0, rng=rng)
        print("Mode: SMART ONLY (100% LLM steps)")
    elif args.dumb_only:
        scheduler = Scheduler(strategy="fixed_ratio", smart_ratio=0.0, rng=rng)
        print("Mode: DUMB ONLY (100% random steps) — baseline")
    else:
        scheduler = Scheduler(
            strategy=args.strategy,
            smart_ratio=args.smart_ratio,
            rng=rng,
        )
        print(f"Mode: {args.strategy}  (smart ratio={args.smart_ratio})")

    print(f"Seed: {args.seed}\n")

    result = run(scheduler=scheduler, seed=args.seed, verbose=not args.quiet)

    print("\n── Best program found ────────────────────────────")
    print(result.best_code)


if __name__ == "__main__":
    main()
