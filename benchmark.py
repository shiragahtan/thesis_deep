"""
benchmark.py
------------
Fixed, reproducible benchmark task: sorting algorithm optimization.

Why sorting?
  - Deterministic: same inputs every run → results are reproducible
  - Gradable: correctness is binary per test case, speed gives a continuous signal
  - Simple enough to demonstrate the loop, rich enough to be non-trivial
  - Analogous to AlphaEvolve's matrix multiplication task

The evaluator returns a score 0–100:
  - Correctness accounts for 70 points  (did it sort correctly?)
  - Speed accounts for 30 points        (how fast relative to Python's built-in?)
"""

from typing import Optional, List
import random
import signal
import time
import traceback
from config import SEED, NUM_TEST_CASES


# ── Timeout helper (Unix/macOS only) ─────────────────────────────────────────

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError("evaluation timed out")


# ── Generate fixed test cases once, seeded ────────────────────────────────────

def _generate_test_cases(seed: int = SEED, n: int = NUM_TEST_CASES) -> List[List[int]]:
    """Return a fixed list of input arrays. Same seed → same cases every run."""
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        size = rng.randint(10, 200)
        arr  = [rng.randint(-1000, 1000) for _ in range(size)]
        cases.append(arr)
    return cases


# Fixed at import time — never changes between runs
TEST_CASES: List[List[int]] = _generate_test_cases()
EXPECTED:   List[List[int]] = [sorted(tc) for tc in TEST_CASES]

# Baseline: measure Python's built-in sort speed once
def _baseline_time() -> float:
    start = time.perf_counter()
    for tc in TEST_CASES:
        sorted(tc)
    return time.perf_counter() - start

BASELINE_TIME: float = _baseline_time()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class EvaluationResult:
    """Holds the full result of evaluating one program."""
    def __init__(self, score: float, correct: int, total: int,
                 speedup: float, error: Optional[str], trace: Optional[str]):
        self.score   = score        # 0–100
        self.correct = correct      # number of test cases passed
        self.total   = total        # total test cases
        self.speedup = speedup      # ratio vs Python built-in (>1 = faster)
        self.error   = error        # last exception message (if any)
        self.trace   = trace        # last traceback (if any)

    def __repr__(self):
        return (f"EvaluationResult(score={self.score:.1f}, "
                f"correct={self.correct}/{self.total}, "
                f"speedup={self.speedup:.2f}x, "
                f"error={self.error!r})")

    def failure_evidence(self) -> str:
        """
        Returns a concise failure report fed to the LLM as context.
        This is E(xₜ) in the thesis formulation:
            xₜ₊₁ = LLM(xₜ, E(xₜ), score(xₜ))
        """
        lines = [f"Score: {self.score:.1f}/100"]
        lines.append(f"Correctness: {self.correct}/{self.total} test cases passed")
        lines.append(f"Speed: {self.speedup:.2f}x vs Python built-in")
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.trace:
            lines.append(f"Traceback (last call):\n{self.trace}")
        return "\n".join(lines)


def evaluate(code: str, timeout: float = 5.0) -> EvaluationResult:
    """
    Run `code` on all fixed test cases and return an EvaluationResult.

    The code must define a function called `sort_array(arr: list) -> list`.
    We run it in a restricted namespace and time it.
    A hard timeout (default 5 s) kills infinite loops via SIGALRM.
    """
    namespace: dict = {}

    # ── 1. compile + exec ─────────────────────────────────────────────────────
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return EvaluationResult(
            score=0.0, correct=0, total=len(TEST_CASES),
            speedup=0.0,
            error=f"Compilation error: {exc}",
            trace=traceback.format_exc(limit=5),
        )

    fn = namespace.get("sort_array")
    if fn is None:
        return EvaluationResult(
            score=0.0, correct=0, total=len(TEST_CASES),
            speedup=0.0,
            error="Function 'sort_array' not found in generated code.",
            trace=None,
        )

    # ── 2. correctness + speed (with timeout) ─────────────────────────────────
    correct = 0
    last_error: Optional[str] = None
    last_trace: Optional[str] = None

    # Set a hard alarm so infinite-loop mutations don't hang the process
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(max(1, timeout)))
    start = time.perf_counter()

    try:
        for tc, expected in zip(TEST_CASES, EXPECTED):
            try:
                result = fn(list(tc))      # pass a copy so tc is not mutated
                if result == expected:
                    correct += 1
            except _TimeoutError:
                raise                      # propagate to outer handler
            except Exception as exc:
                last_error = str(exc)
                last_trace = traceback.format_exc(limit=5)
    except _TimeoutError:
        elapsed = time.perf_counter() - start
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return EvaluationResult(
            score=0.0, correct=correct, total=len(TEST_CASES),
            speedup=0.0,
            error=f"Timeout after {elapsed:.1f}s (infinite loop?)",
            trace=None,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.perf_counter() - start
    speedup = BASELINE_TIME / elapsed if elapsed > 0 else 0.0

    # ── 3. scoring ─────────────────────────────────────────────────────────────
    correctness_score = 70 * (correct / len(TEST_CASES))
    speed_score       = min(30, 30 * speedup)   # capped at 30 pts
    total_score       = correctness_score + speed_score

    return EvaluationResult(
        score=round(total_score, 2),
        correct=correct,
        total=len(TEST_CASES),
        speedup=round(speedup, 3),
        error=last_error,
        trace=last_trace,
    )


# ── Initial seed program ───────────────────────────────────────────────────────

SEED_PROGRAM = '''\
def sort_array(arr: list) -> list:
    """Baseline: selection sort — correct but O(n²)."""
    arr = list(arr)
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
'''
