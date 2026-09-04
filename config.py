"""
config.py
---------
Central configuration for the LLM-Guided Evolutionary Search experiment.
All parameters live here — change them in one place, they propagate everywhere.
"""

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42                   # fixed seed for all random operations

# ── Population ────────────────────────────────────────────────────────────────
POPULATION_SIZE  = 10       # number of programs kept alive at any time
TOP_K_PARENTS    = 3        # how many top programs are selected as parents
NUM_PROPOSERS    = 5        # candidates generated per mutation step (pick best)

# ── Loop ──────────────────────────────────────────────────────────────────────
MAX_GENERATIONS  = 50       # stop after this many generations
TARGET_SCORE     = 95       # stop early if any program reaches this score

# ── Step scheduling ───────────────────────────────────────────────────────────
SMART_STEP_RATIO = 0.7      # fraction of steps that use the LLM (smart step)
                             # remaining (1 - ratio) use random mutation (dumb step)

# ── LLM (Smart Step) ──────────────────────────────────────────────────────────
LLM_MODEL        = "claude-haiku-4-5-20251001"   # fast + cheap for many calls
LLM_MAX_TOKENS   = 1024
LLM_TEMPERATURE  = 0.8      # some creativity in mutations

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_NAME   = "sorting"   # task to solve (see benchmark.py)
NUM_TEST_CASES   = 20          # fixed test cases, same every run

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR      = "results"
LOG_EVERY        = 5           # print progress every N generations
