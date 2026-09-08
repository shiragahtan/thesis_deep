"""
benchmark_strings.py
--------------------
Benchmark task: substring search (pattern matching).

Why string search?
  - Seed: naive O(n*m) brute-force search
  - Target: KMP (Knuth-Morris-Pratt) O(n+m) — classic algorithm
  - Tests a completely different domain: string processing vs numerical sorting
  - The LLM can reason from "slow on long strings" → suggest prefix-table approach

Scoring: 70% correctness + 30% speed vs naive baseline
The code must define: find_pattern(text: str, pattern: str) -> list[int]
  Return list of all start indices where pattern appears in text (0-indexed).
"""

from typing import Optional, List
import random
import time
import traceback
import signal
from config import SEED
from benchmark import EvaluationResult

# ── Fixed test cases ──────────────────────────────────────────────────────────

def _random_string(length: int, alphabet: str, rng: random.Random) -> str:
    return ''.join(rng.choice(alphabet) for _ in range(length))

def _naive_search(text: str, pattern: str) -> List[int]:
    """Correct O(n*m) — used to generate expected outputs."""
    results = []
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            results.append(i)
    return results

def _generate_test_cases(seed: int = SEED):
    rng = random.Random(seed)
    cases = []
    for _ in range(15):
        alphabet = rng.choice(['ab', 'abc', 'abcd'])
        text_len = rng.randint(200, 2000)
        pat_len  = rng.randint(3, 15)
        text    = _random_string(text_len, alphabet, rng)
        pattern = _random_string(pat_len, alphabet, rng)
        cases.append((text, pattern))
    return cases

TEST_CASES_STRINGS = _generate_test_cases()
EXPECTED_STRINGS   = [_naive_search(t, p) for t, p in TEST_CASES_STRINGS]

def _fast_search(text: str, pattern: str) -> List[int]:
    """Fast baseline using Python's built-in str.find (implemented in C)."""
    results = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        results.append(pos)
        start = pos + 1
    return results

# Baseline = built-in str.find (C-speed), so pure Python seed scores low
def _baseline_time_strings() -> float:
    start = time.perf_counter()
    for text, pattern in TEST_CASES_STRINGS:
        _fast_search(text, pattern)
    return time.perf_counter() - start

BASELINE_TIME_STRINGS: float = _baseline_time_strings()

# ── Timeout helper ────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()

# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate_strings(code: str, timeout: float = 10.0) -> EvaluationResult:
    """
    Run code on all string test cases. Must define find_pattern(text, pattern) -> list[int].
    Score = 70*(correct/total) + min(30, 30*speedup)
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return EvaluationResult(0.0, 0, len(TEST_CASES_STRINGS), 0.0,
                                f"Compilation error: {exc}", traceback.format_exc(limit=5))

    fn = namespace.get("find_pattern")
    if fn is None:
        return EvaluationResult(0.0, 0, len(TEST_CASES_STRINGS), 0.0,
                                "Function 'find_pattern' not found.", None)

    correct = 0
    last_error: Optional[str] = None
    last_trace: Optional[str] = None

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(max(1, timeout)))
    start = time.perf_counter()
    try:
        for (text, pattern), expected in zip(TEST_CASES_STRINGS, EXPECTED_STRINGS):
            try:
                result = fn(text, pattern)
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
        return EvaluationResult(0.0, correct, len(TEST_CASES_STRINGS), 0.0,
                                f"Timeout after {elapsed:.1f}s", None)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.perf_counter() - start
    speedup = BASELINE_TIME_STRINGS / elapsed if elapsed > 0 else 0.0
    score = 70 * (correct / len(TEST_CASES_STRINGS)) + min(30, 30 * speedup)

    return EvaluationResult(round(score, 2), correct, len(TEST_CASES_STRINGS),
                            round(speedup, 3), last_error, last_trace)


# ── Seed program ──────────────────────────────────────────────────────────────

SEED_PROGRAM_STRINGS = '''\
def find_pattern(text: str, pattern: str) -> list:
    """Baseline: naive brute-force O(n*m) string search."""
    results = []
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            results.append(i)
    return results
'''

if __name__ == "__main__":
    result = evaluate_strings(SEED_PROGRAM_STRINGS)
    print(f"String search benchmark — seed program:")
    print(f"  Score:   {result.score:.1f}/100")
    print(f"  Correct: {result.correct}/{result.total}")
    print(f"  Speedup: {result.speedup:.2f}x vs naive baseline")
    print(f"  Sample: text_len={len(TEST_CASES_STRINGS[0][0])}, pat='{TEST_CASES_STRINGS[0][1]}'")
