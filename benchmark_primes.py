"""
benchmark_primes.py
-------------------
Benchmark task: prime number generation (Sieve of Eratosthenes).

Why primes?
  - Seed: naive O(n²) trial division — correct but very slow
  - Target: Sieve of Eratosthenes O(n log log n) — classic algorithm
  - The LLM can reason from "slow" signal → suggest sieve structure
  - Completely different domain from sorting → tests generalization

Scoring: 70% correctness + 30% speed vs naive baseline
The code must define: get_primes(n: int) -> list[int]
  Return all primes p where 2 <= p <= n, in ascending order.
"""

from typing import Optional, List
import random
import time
import traceback
from config import SEED

# ── Fixed test cases ──────────────────────────────────────────────────────────

def _generate_test_cases(seed: int = SEED) -> List[int]:
    """Return fixed upper bounds N to find primes up to."""
    rng = random.Random(seed)
    cases = []
    for _ in range(10):
        n = rng.randint(100, 5000)
        cases.append(n)
    return cases

def _naive_primes(n: int) -> List[int]:
    """Correct O(n * sqrt(n)) — used to generate expected outputs."""
    primes = []
    for candidate in range(2, n + 1):
        is_prime = True
        for d in range(2, int(candidate**0.5) + 1):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
    return primes

def _sieve(n: int) -> List[int]:
    """Sieve of Eratosthenes — the fast target algorithm."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

TEST_CASES_PRIMES: List[int]      = _generate_test_cases()
EXPECTED_PRIMES:  List[List[int]] = [_naive_primes(n) for n in TEST_CASES_PRIMES]

# Baseline = SIEVE (fast), so trial division seed scores low on speed
def _baseline_time_primes() -> float:
    start = time.perf_counter()
    for n in TEST_CASES_PRIMES:
        _sieve(n)
    return time.perf_counter() - start

BASELINE_TIME_PRIMES: float = _baseline_time_primes()

# ── Evaluator ─────────────────────────────────────────────────────────────────

import signal

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()

from benchmark import EvaluationResult

def evaluate_primes(code: str, timeout: float = 10.0) -> EvaluationResult:
    """
    Run code on all prime test cases. Must define get_primes(n) -> list[int].
    Score = 70*(correct/total) + min(30, 30*speedup)
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return EvaluationResult(0.0, 0, len(TEST_CASES_PRIMES), 0.0,
                                f"Compilation error: {exc}", traceback.format_exc(limit=5))

    fn = namespace.get("get_primes")
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES_PRIMES), 0.0,
                                "Function 'get_primes' not found.", None)

    correct = 0
    last_error: Optional[str] = None
    last_trace: Optional[str] = None

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(max(1, timeout)))
    start = time.perf_counter()
    try:
        for n, expected in zip(TEST_CASES_PRIMES, EXPECTED_PRIMES):
            try:
                result = fn(n)
                if sorted(result) == expected:
                    correct += 1
            except _TimeoutError:
                raise
            except Exception as exc:
                last_error = str(exc)
                last_trace = traceback.format_exc(limit=5)
    except _TimeoutError:
        elapsed = time.perf_counter() - start
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        return EvaluationResult(0.0, correct, len(TEST_CASES_PRIMES), 0.0,
                                f"Timeout after {elapsed:.1f}s", None)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.perf_counter() - start
    speedup = BASELINE_TIME_PRIMES / elapsed if elapsed > 0 else 0.0
    score = 70 * (correct / len(TEST_CASES_PRIMES)) + min(30, 30 * speedup)

    return EvaluationResult(round(score, 2), correct, len(TEST_CASES_PRIMES),
                            round(speedup, 3), last_error, last_trace)


# ── Seed program ──────────────────────────────────────────────────────────────

SEED_PROGRAM_PRIMES = '''\
def get_primes(n: int) -> list:
    """Baseline: trial division O(n * sqrt(n)) — correct but slow."""
    primes = []
    for candidate in range(2, n + 1):
        is_prime = True
        for d in range(2, int(candidate**0.5) + 1):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
    return primes
'''

if __name__ == "__main__":
    result = evaluate_primes(SEED_PROGRAM_PRIMES)
    print(f"Primes benchmark — seed program:")
    print(f"  Score:   {result.score:.1f}/100")
    print(f"  Correct: {result.correct}/{result.total}")
    print(f"  Speedup: {result.speedup:.2f}x vs naive baseline")
    print(f"  Sample N values: {TEST_CASES_PRIMES[:5]}")
