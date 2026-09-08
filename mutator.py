"""
mutator.py
----------
Implements the two mutation strategies at the heart of the thesis:

  DUMB STEP  →  xₜ₊₁ = Mutate_random(xₜ)
                Random syntactic change. No knowledge of why the program failed.

  SMART STEP →  xₜ₊₁ = LLM(xₜ, E(xₜ), score(xₜ))
                The LLM reads the code AND the failure evidence before proposing
                a fix. This is the thesis contribution.

Both functions return a new code string (or None if mutation fails).
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import os
import random
import re
from benchmark import EvaluationResult
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


# ── Smart Step (LLM-guided) ────────────────────────────────────────────────────

# Lazy client — only instantiated when smart_step is actually called.
# This lets dumb-only runs import mutator without a GROQ_API_KEY.
_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from groq import Groq
        _CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _CLIENT

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Python programmer helping optimize an algorithm.
You will be shown:
  1. The current code
  2. How it performed (score, correctness, speed)
  3. Any error or failure evidence

Your task: propose ONE improved version of the function `{fn_hint}`.

Rules:
- Return ONLY the Python code block, no explanation.
- The function signature must stay exactly the same.
- Do not use Python's built-in sorting functions.
- Make a targeted change based on the failure evidence.
"""

# Default: sorting benchmark
_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(fn_hint="sort_array(arr: list) -> list")

def smart_step(code: str, eval_result: EvaluationResult) -> Optional[str]:
    """
    Call the LLM with the current code + failure evidence.
    Returns the improved code string, or None if the call fails.

    This implements:  xₜ₊₁ = LLM(xₜ, E(xₜ), score(xₜ))
    """
    user_message = f"""\
## Current code
```python
{code}
```

## Failure evidence
{eval_result.failure_evidence()}

Please propose an improved version of `sort_array`.
"""
    import time as _time
    for attempt in range(4):   # retry up to 4 times on transient errors
        try:
            response = _get_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            raw = response.choices[0].message.content
            return _extract_code(raw)
        except Exception as exc:
            msg = str(exc)
            is_transient = ("429" in msg or "Connection error" in msg
                            or "timeout" in msg.lower() or "502" in msg
                            or "503" in msg)
            if is_transient and attempt < 3:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                print(f"  [smart_step] Transient error (attempt {attempt+1}), retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"  [smart_step] LLM call failed: {exc}")
                return None
    return None


def _extract_code(text: str) -> str:
    """Extract the first Python code block from the LLM response."""
    # Try ```python ... ``` block first
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to ``` ... ```
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return raw text if no code block found
    return text.strip()


# ── Dumb Step (random mutation) ────────────────────────────────────────────────

def dumb_step(code: str, rng: random.Random) -> Optional[str]:
    """
    Apply a random syntactic mutation to the code.
    No knowledge of why the program failed.

    This implements:  xₜ₊₁ = Mutate_random(xₜ)

    Mutations (chosen randomly):
      1. Swap two adjacent lines
      2. Replace a numeric literal with a nearby value
      3. Swap a comparison operator
      4. Duplicate a line
      5. Delete a non-essential line
      6. Insert a sorting primitive (NEW — structural mutation)
      7. Wrap inner loop with early-exit guard (NEW — structural mutation)
      8. Rename a loop variable (NEW — cosmetic, tests population diversity)
    """
    mutations = [
        _swap_lines,
        _perturb_number,
        _swap_operator,
        _duplicate_line,
        _delete_line,
        _insert_primitive,
        _add_early_exit,
        _rename_variable,
    ]
    chosen = rng.choice(mutations)
    try:
        result = chosen(code, rng)
        return result if result and result.strip() else None
    except Exception:
        return None


def _swap_lines(code: str, rng: random.Random) -> Optional[str]:
    lines = code.splitlines()
    if len(lines) < 3:
        return None
    i = rng.randint(1, len(lines) - 2)
    lines[i], lines[i + 1] = lines[i + 1], lines[i]
    return "\n".join(lines)


def _perturb_number(code: str, rng: random.Random) -> Optional[str]:
    numbers = [(m.start(), m.group()) for m in re.finditer(r'\b\d+\b', code)]
    if not numbers:
        return None
    pos, num_str = rng.choice(numbers)
    num = int(num_str)
    delta = rng.choice([-1, 1, -2, 2])
    new_num = max(0, num + delta)
    return code[:pos] + str(new_num) + code[pos + len(num_str):]


def _swap_operator(code: str, rng: random.Random) -> Optional[str]:
    pairs = [('<', '>'), ('<=', '>='), ('+', '-'), ('*', '//')]
    candidates = [(a, b) for a, b in pairs if a in code]
    if not candidates:
        return None
    a, b = rng.choice(candidates)
    if rng.random() < 0.5:
        return code.replace(a, b, 1)
    elif b in code:
        return code.replace(b, a, 1)
    return None


def _duplicate_line(code: str, rng: random.Random) -> Optional[str]:
    lines = code.splitlines()
    if len(lines) < 2:
        return None
    i = rng.randint(0, len(lines) - 1)
    lines.insert(i + 1, lines[i])
    return "\n".join(lines)


def _delete_line(code: str, rng: random.Random) -> Optional[str]:
    lines = code.splitlines()
    # Only delete non-def, non-return, non-empty lines
    deletable = [i for i, l in enumerate(lines)
                 if l.strip() and not l.strip().startswith(('def ', 'return', '"""', "'''"))]
    if not deletable:
        return None
    i = rng.choice(deletable)
    del lines[i]
    return "\n".join(lines)


# ── Structural mutations (stronger, but still blind) ──────────────────────────

_PRIMITIVES = [
    # Insertion sort kernel
    ("insertion_sort", """\
def _insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
"""),
    # Merge helper
    ("merge_sort", """\
def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def _merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    return _merge(_merge_sort(arr[:mid]), _merge_sort(arr[mid:]))
"""),
    # Quicksort — iterative to avoid recursion limit on arrays up to 200
    ("quick_sort", """\
def _quick_sort(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        p = i + 1
        stack.append((lo, p - 1))
        stack.append((p + 1, hi))
"""),
    # Heap sort — iterative heapify, no recursion
    ("heap_sort", """\
def _heapify(arr, n, i):
    while True:
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest
"""),
]

_PRIMITIVE_WRAPPERS = {
    "insertion_sort": "    arr = list(arr)\n    return _insertion_sort(arr)\n",
    "merge_sort":     "    return _merge_sort(list(arr))\n",
    "quick_sort":     "    arr = list(arr)\n    _quick_sort(arr)\n    return arr\n",
    "heap_sort": (
        "    arr = list(arr)\n"
        "    n = len(arr)\n"
        "    for i in range(n // 2 - 1, -1, -1):\n"
        "        _heapify(arr, n, i)\n"
        "    for i in range(n - 1, 0, -1):\n"
        "        arr[0], arr[i] = arr[i], arr[0]\n"
        "        _heapify(arr, i, 0)\n"
        "    return arr\n"
    ),
}


def _insert_primitive(code: str, rng: random.Random) -> Optional[str]:
    """
    Replace the body of sort_array with a call to a known-good sorting primitive.
    This is a 'strong dumb step' — structurally meaningful but still random (no LLM).
    """
    name, helper = rng.choice(_PRIMITIVES)
    wrapper = _PRIMITIVE_WRAPPERS[name]
    new_code = helper + "\ndef sort_array(arr: list) -> list:\n" + wrapper
    return new_code


def _add_early_exit(code: str, rng: random.Random) -> Optional[str]:
    """
    Add an early-exit check at the top of sort_array: if len <= 1, return immediately.
    Harmless optimization that often helps speed on already-short arrays.
    """
    if "if len(arr) <= 1" in code:
        return None  # already there
    lines = code.splitlines()
    # Find the line after 'def sort_array'
    for i, line in enumerate(lines):
        if line.strip().startswith("def sort_array"):
            indent = "    "
            lines.insert(i + 1, f"{indent}if len(arr) <= 1:\n{indent}    return list(arr)")
            return "\n".join(lines)
    return None


def _rename_variable(code: str, rng: random.Random) -> Optional[str]:
    """
    Rename a loop variable (i→ii, j→jj, etc.) — tests that population
    can still accept semantically equivalent programs.
    """
    pairs = [('\\bi\\b', 'ii'), ('\\bj\\b', 'jj'), ('\\bk\\b', 'kk'),
             ('\\bmin_idx\\b', 'min_i'), ('\\bpivot\\b', 'piv')]
    rng.shuffle(pairs)
    for pattern, replacement in pairs:
        new_code = re.sub(pattern, replacement, code)
        if new_code != code:
            return new_code
    return None
