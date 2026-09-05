"""
benchmark_matrix.py
-------------------
Second benchmark task: matrix multiplication optimization.

Why matrix multiplication?
  - This is exactly what AlphaEvolve optimized (Strassen's algorithm)
  - Correctness is exact (element-wise equality)
  - Speed is the primary signal (number of operations matters)
  - Directly analogous to the real-world AlphaEvolve result

The evaluator returns a score 0–100:
  - Correctness: 60 points  (must match numpy's result exactly, within tolerance)
  - Speed:       40 points  (how fast vs naive O(n³) implementation)

The seed program is a correct but naive O(n³) triple-loop multiplication.
The search should discover Strassen-like optimizations.

Note: we use small matrices (4x4, 8x8) so the overhead of Python loops
is measurable but the test cases complete quickly.
"""

from typing import Optional, List
import random
import time
import traceback
from config import SEED

# ── Generate fixed test cases ─────────────────────────────────────────────────

def _random_matrix(n: int, rng: random.Random) -> List[List[float]]:
    return [[round(rng.uniform(-10, 10), 2) for _ in range(n)] for _ in range(n)]

def _generate_test_cases(seed: int = SEED) -> List[tuple]:
    """Return fixed (A, B, expected_C) triples for matrix sizes 2,4,8."""
    rng = random.Random(seed)
    cases = []
    for _ in range(5):    # 5 cases per size
        for n in [2, 4, 8]:
            A = _random_matrix(n, rng)
            B = _random_matrix(n, rng)
            C = _naive_multiply(A, B)
            cases.append((A, B, C))
    return cases

def _naive_multiply(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def _matrices_close(C1, C2, tol=1e-6) -> bool:
    """Element-wise comparison within tolerance."""
    for row1, row2 in zip(C1, C2):
        for a, b in zip(row1, row2):
            if abs(a - b) > tol:
                return False
    return True

# Fixed test cases — same every run
TEST_CASES_MATRIX = _generate_test_cases()

# Baseline speed: naive O(n³)
def _baseline_time_matrix() -> float:
    start = time.perf_counter()
    for A, B, _ in TEST_CASES_MATRIX:
        _naive_multiply(A, B)
    return time.perf_counter() - start

BASELINE_TIME_MATRIX: float = _baseline_time_matrix()


# ── EvaluationResult (same structure as benchmark.py) ────────────────────────

from benchmark import EvaluationResult


def evaluate_matrix(code: str) -> EvaluationResult:
    """
    Run `code` on all matrix test cases.
    The code must define: matmul(A, B) -> list[list[float]]
    Score = 60*(correct/total) + min(40, 40*speedup)
    """
    namespace: dict = {}

    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return EvaluationResult(
            score=0.0, correct=0, total=len(TEST_CASES_MATRIX),
            speedup=0.0,
            error=f"Compilation error: {exc}",
            trace=traceback.format_exc(limit=5),
        )

    fn = namespace.get("matmul")
    if fn is None:
        return EvaluationResult(
            score=0.0, correct=0, total=len(TEST_CASES_MATRIX),
            speedup=0.0,
            error="Function 'matmul' not found in generated code.",
            trace=None,
        )

    correct = 0
    last_error: Optional[str] = None
    last_trace: Optional[str] = None
    start = time.perf_counter()

    for A, B, expected in TEST_CASES_MATRIX:
        try:
            result = fn([row[:] for row in A], [row[:] for row in B])
            if _matrices_close(result, expected):
                correct += 1
        except Exception as exc:
            last_error = str(exc)
            last_trace = traceback.format_exc(limit=5)

    elapsed = time.perf_counter() - start
    speedup = BASELINE_TIME_MATRIX / elapsed if elapsed > 0 else 0.0

    correctness_score = 60 * (correct / len(TEST_CASES_MATRIX))
    speed_score       = min(40, 40 * speedup)
    total_score       = correctness_score + speed_score

    return EvaluationResult(
        score=round(total_score, 2),
        correct=correct,
        total=len(TEST_CASES_MATRIX),
        speedup=round(speedup, 3),
        error=last_error,
        trace=last_trace,
    )


# ── Seed program ──────────────────────────────────────────────────────────────

SEED_PROGRAM_MATRIX = '''\
def matmul(A: list, B: list) -> list:
    """Baseline: naive O(n³) matrix multiplication."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
'''

if __name__ == "__main__":
    result = evaluate_matrix(SEED_PROGRAM_MATRIX)
    print(f"Matrix benchmark — seed program:")
    print(f"  Score:   {result.score:.1f}/100")
    print(f"  Correct: {result.correct}/{result.total}")
    print(f"  Speedup: {result.speedup:.2f}x vs naive baseline")
