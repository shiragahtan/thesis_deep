"""
evaluators.py
-------------
Multiple evaluator variants for ablation experiments.

The default evaluator (in benchmark.py) combines correctness + speed:
    score = 70 * (correct/total) + min(30, 30 * speedup)

But what if we only reward correctness? Or only speed?
These variants let us study how the choice of evaluation signal
affects the evolutionary dynamics — and how the LLM uses that signal.

Evaluator variants:
  - combined      : default (70% correctness + 30% speed)  [same as benchmark.py]
  - correctness   : 100% correctness, ignore speed
  - speed         : 100% speed, only on correct programs
  - strict_speed  : correctness is a hard requirement; speed is the full score

Usage:
    from evaluators import get_evaluator
    evaluate = get_evaluator("correctness")
    result = evaluate(code)
"""

from typing import Optional, Callable
import random
import time
import traceback

from benchmark import (
    TEST_CASES, EXPECTED, BASELINE_TIME,
    EvaluationResult,
)


# ── Shared execution helper ────────────────────────────────────────────────────

def _run_code(code: str):
    """
    Compile and exec `code`, return (fn, error, trace).
    fn is the `sort_array` function, or None on failure.
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return None, f"Compilation error: {exc}", traceback.format_exc(limit=5)

    fn = namespace.get("sort_array")
    if fn is None:
        return None, "Function 'sort_array' not found.", None

    return fn, None, None


def _execute(fn) -> tuple:
    """
    Run fn on all test cases.
    Returns (correct_count, elapsed_seconds, last_error, last_trace).
    """
    correct = 0
    last_error: Optional[str] = None
    last_trace: Optional[str] = None
    start = time.perf_counter()

    for tc, expected in zip(TEST_CASES, EXPECTED):
        try:
            result = fn(list(tc))
            if result == expected:
                correct += 1
        except Exception as exc:
            last_error = str(exc)
            last_trace = traceback.format_exc(limit=5)

    elapsed = time.perf_counter() - start
    return correct, elapsed, last_error, last_trace


# ── Evaluator variants ─────────────────────────────────────────────────────────

def evaluate_combined(code: str) -> EvaluationResult:
    """
    Default: 70 pts correctness + 30 pts speed.
    Identical to benchmark.evaluate() — included here for completeness.
    """
    fn, err, trace = _run_code(code)
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES), 0.0, err, trace)

    correct, elapsed, last_err, last_trace = _execute(fn)
    speedup = BASELINE_TIME / elapsed if elapsed > 0 else 0.0

    score = 70 * (correct / len(TEST_CASES)) + min(30, 30 * speedup)
    return EvaluationResult(
        score=round(score, 2),
        correct=correct,
        total=len(TEST_CASES),
        speedup=round(speedup, 3),
        error=last_err,
        trace=last_trace,
    )


def evaluate_correctness_only(code: str) -> EvaluationResult:
    """
    100 pts correctness only. Speed is ignored.
    Use this to study: does the LLM converge faster when only accuracy matters?
    """
    fn, err, trace = _run_code(code)
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES), 0.0, err, trace)

    correct, elapsed, last_err, last_trace = _execute(fn)
    speedup = BASELINE_TIME / elapsed if elapsed > 0 else 0.0

    score = 100.0 * (correct / len(TEST_CASES))
    return EvaluationResult(
        score=round(score, 2),
        correct=correct,
        total=len(TEST_CASES),
        speedup=round(speedup, 3),
        error=last_err,
        trace=last_trace,
    )


def evaluate_speed_only(code: str) -> EvaluationResult:
    """
    Speed is the full score (0–100), BUT only for programs that pass all tests.
    Incorrect programs score 0 regardless of speed.

    Use this to study: can the search discover fast algorithms when given
    only a speed signal — with correctness as a hard gate?
    """
    fn, err, trace = _run_code(code)
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES), 0.0, err, trace)

    correct, elapsed, last_err, last_trace = _execute(fn)

    if correct < len(TEST_CASES):
        # Hard gate: incorrect programs get 0
        return EvaluationResult(
            score=0.0,
            correct=correct,
            total=len(TEST_CASES),
            speedup=0.0,
            error=last_err or f"Only {correct}/{len(TEST_CASES)} correct",
            trace=last_trace,
        )

    speedup = BASELINE_TIME / elapsed if elapsed > 0 else 0.0
    score = min(100.0, 100.0 * speedup)   # 1x speedup = 100 pts
    return EvaluationResult(
        score=round(score, 2),
        correct=correct,
        total=len(TEST_CASES),
        speedup=round(speedup, 3),
        error=None,
        trace=None,
    )


def evaluate_strict_speed(code: str) -> EvaluationResult:
    """
    Like speed_only but score scales linearly up to 5x baseline:
        score = min(100, 20 * speedup)
    This gives finer resolution at lower speedups and rewards incremental gains.
    """
    fn, err, trace = _run_code(code)
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES), 0.0, err, trace)

    correct, elapsed, last_err, last_trace = _execute(fn)

    if correct < len(TEST_CASES):
        return EvaluationResult(
            score=0.0,
            correct=correct,
            total=len(TEST_CASES),
            speedup=0.0,
            error=last_err or f"Only {correct}/{len(TEST_CASES)} correct",
            trace=last_trace,
        )

    speedup = BASELINE_TIME / elapsed if elapsed > 0 else 0.0
    score = min(100.0, 20.0 * speedup)   # 5x speedup = 100 pts
    return EvaluationResult(
        score=round(score, 2),
        correct=correct,
        total=len(TEST_CASES),
        speedup=round(speedup, 3),
        error=None,
        trace=None,
    )


# ── Registry ───────────────────────────────────────────────────────────────────

EVALUATORS: dict = {
    "combined":        evaluate_combined,
    "correctness":     evaluate_correctness_only,
    "speed":           evaluate_speed_only,
    "strict_speed":    evaluate_strict_speed,
}


def get_evaluator(name: str) -> Callable:
    """Return an evaluator function by name."""
    if name not in EVALUATORS:
        raise ValueError(f"Unknown evaluator '{name}'. Choose from: {list(EVALUATORS)}")
    return EVALUATORS[name]


if __name__ == "__main__":
    # Quick sanity check on the seed program
    from benchmark import SEED_PROGRAM
    print("Evaluator sanity check on selection sort:\n")
    for name, fn in EVALUATORS.items():
        r = fn(SEED_PROGRAM)
        print(f"  {name:<18} score={r.score:.1f}  correct={r.correct}/{r.total}  speedup={r.speedup:.2f}x")
