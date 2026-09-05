"""
loop.py
-------
The main evolutionary loop.

Each generation:
  1. Select a parent from the top-K population
  2. Decide smart vs dumb step (via Scheduler)
  3. Generate NUM_PROPOSERS candidate mutations (multiple proposers)
  4. Evaluate all candidates on the fixed benchmark
  5. Accept the best candidate if it beats the worst in the population
  6. Log progress

This is the full implementation of:
  xₜ₊₁ = LLM(xₜ, E(xₜ), score(xₜ))   [smart step]
  xₜ₊₁ = Mutate_random(xₜ)             [dumb step]
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import json
import os
import random
import time
from dataclasses import dataclass, asdict

import config as _cfg
from benchmark import evaluate, SEED_PROGRAM
from mutator import smart_step, dumb_step
from population import Population, Program
from scheduler import Scheduler


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GenerationLog:
    generation:   int
    best_score:   float
    mean_score:   float
    diversity:    float
    step_type:    str        # "smart" or "dumb"
    accepted:     bool       # was the best proposer accepted into the population?
    proposers_tried: int
    elapsed_sec:  float


@dataclass
class RunResult:
    config:       dict
    generations:  List[GenerationLog]
    best_code:    str
    best_score:   float
    total_evaluations: int
    reached_target:    bool
    elapsed_sec:  float


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(
    scheduler: Optional[Scheduler] = None,
    seed: int = None,
    verbose: bool = True,
    max_generations: Optional[int] = None,
    num_proposers: Optional[int] = None,
    population_size: Optional[int] = None,
    top_k: Optional[int] = None,
) -> RunResult:
    """
    Run the full evolutionary search and return a RunResult.

    Args:
        scheduler:        Scheduler instance (default: fixed_ratio)
        seed:             random seed for reproducibility
        verbose:          print progress logs
        max_generations:  override config.MAX_GENERATIONS
        num_proposers:    override config.NUM_PROPOSERS
        population_size:  override config.POPULATION_SIZE
        top_k:            override config.TOP_K_PARENTS
    """
    # Read from live config (allows runtime overrides) then apply call-site overrides
    _seed        = seed if seed is not None else _cfg.SEED
    _max_gen     = max_generations  if max_generations  is not None else _cfg.MAX_GENERATIONS
    _proposers   = num_proposers    if num_proposers    is not None else _cfg.NUM_PROPOSERS
    _pop_size    = population_size  if population_size  is not None else _cfg.POPULATION_SIZE
    _top_k       = top_k            if top_k            is not None else _cfg.TOP_K_PARENTS

    rng       = random.Random(_seed)
    scheduler = scheduler or Scheduler(rng=rng)
    pop       = Population(max_size=_pop_size, top_k=min(_top_k, _pop_size))

    # ── seed the population with the baseline program ─────────────────────────
    seed_eval = evaluate(SEED_PROGRAM)
    pop.add(Program(code=SEED_PROGRAM, score=seed_eval.score,
                    eval_result=seed_eval, generation=0))
    if verbose:
        print(f"[Gen 0] Seed program score: {seed_eval.score:.1f}/100")
        print(f"        {seed_eval}")

    logs: List[GenerationLog] = []
    total_evals = 1
    run_start   = time.perf_counter()

    for gen in range(1, _max_gen + 1):
        gen_start = time.perf_counter()

        # ── 1. select parent ─────────────────────────────────────────────────
        parent = pop.select_parent(rng)

        # ── 2. decide step type ───────────────────────────────────────────────
        diversity   = pop.diversity()
        use_smart   = scheduler.use_smart_step(diversity=diversity)
        step_type   = "smart" if use_smart else "dumb"

        # ── 3. generate _proposers candidates ────────────────────────────────
        candidates: List[Tuple[str, object]] = []   # (code, eval_result)

        for _ in range(_proposers):
            if use_smart:
                new_code = smart_step(parent.code, parent.eval_result)
            else:
                new_code = dumb_step(parent.code, rng)

            if new_code is None:
                continue

            result = evaluate(new_code)
            total_evals += 1
            candidates.append((new_code, result))

        # ── 4. pick best candidate ────────────────────────────────────────────
        if not candidates:
            accepted = False
            best_candidate_score = 0.0
        else:
            best_code, best_eval = max(candidates, key=lambda c: c[1].score)
            accepted = pop.add(
                Program(code=best_code, score=best_eval.score,
                        eval_result=best_eval, generation=gen)
            )
            best_candidate_score = best_eval.score

        # ── 5. log ────────────────────────────────────────────────────────────
        scores     = pop.all_scores()
        mean_score = sum(scores) / len(scores)
        elapsed    = time.perf_counter() - gen_start

        log = GenerationLog(
            generation=gen,
            best_score=pop.best().score,
            mean_score=round(mean_score, 2),
            diversity=round(diversity, 2),
            step_type=step_type,
            accepted=accepted,
            proposers_tried=len(candidates),
            elapsed_sec=round(elapsed, 3),
        )
        logs.append(log)

        if verbose and (gen % _cfg.LOG_EVERY == 0 or gen == 1):
            print(f"[Gen {gen:3d}] best={pop.best().score:.1f}  "
                  f"mean={mean_score:.1f}  div={diversity:.1f}  "
                  f"step={step_type}  accepted={accepted}  "
                  f"evals={total_evals}")

        # ── 6. early stop ─────────────────────────────────────────────────────
        if pop.best().score >= _cfg.TARGET_SCORE:
            if verbose:
                print(f"\n✓ Target score {_cfg.TARGET_SCORE} reached at generation {gen}!")
            break

    total_elapsed = time.perf_counter() - run_start
    best = pop.best()

    result = RunResult(
        config=_current_config(),
        generations=logs,
        best_code=best.code,
        best_score=best.score,
        total_evaluations=total_evals,
        reached_target=best.score >= _cfg.TARGET_SCORE,
        elapsed_sec=round(total_elapsed, 2),
    )

    _save_result(result)

    if verbose:
        print(f"\n── Run complete ──────────────────────────────")
        print(f"   Best score:        {best.score:.1f}/100")
        print(f"   Total evaluations: {total_evals}")
        print(f"   Reached target:    {result.reached_target}")
        print(f"   Time:              {total_elapsed:.1f}s")
        print(f"   Scheduler:         {scheduler}")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _current_config() -> dict:
    import config
    return {k: v for k, v in vars(config).items()
            if not k.startswith("_")}


def _save_result(result: RunResult):
    os.makedirs(_cfg.RESULTS_DIR, exist_ok=True)
    timestamp = int(time.time())
    path = os.path.join(_cfg.RESULTS_DIR, f"run_{timestamp}.json")

    # Convert to JSON-serializable dict
    data = {
        "config":            result.config,
        "best_score":        result.best_score,
        "total_evaluations": result.total_evaluations,
        "reached_target":    result.reached_target,
        "elapsed_sec":       result.elapsed_sec,
        "best_code":         result.best_code,
        "generations": [asdict(g) for g in result.generations],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   Results saved → {path}")
