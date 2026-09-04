# LLM-Guided Evolutionary Search — Proof of Concept

This codebase implements the core thesis idea:

> **Replace AlphaEvolve's blind mutation with an LLM that reads failure evidence before proposing a fix.**

---

## The Core Idea

AlphaEvolve's mutation step:
```
xₜ₊₁ = LLM( xₜ )
```
The LLM only sees the code. It is blind to why it failed.

**This thesis proposes:**
```
xₜ₊₁ = LLM( xₜ , E(xₜ) , score(xₜ) )
```
The LLM sees the code **+ execution trace + error log + fitness score**.  
This makes mutation *diagnosis-driven*, not semi-blind.

---

## File Structure

```
code/
├── main.py          # entry point — run experiments from here
├── config.py        # all parameters in one place
├── benchmark.py     # fixed reproducible task (sorting) + evaluator
├── population.py    # manages the population of candidate programs
├── mutator.py       # smart step (LLM) + dumb step (random)
├── scheduler.py     # decides when to use smart vs dumb step
├── loop.py          # the main evolutionary loop
├── requirements.txt
└── results/         # JSON logs saved here after each run
```

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Running Experiments

```bash
# Full run (smart + dumb steps, fixed ratio):
python main.py

# Ablation: smart steps only (upper bound):
python main.py --smart-only

# Ablation: dumb steps only (baseline — equivalent to AlphaEvolve):
python main.py --dumb-only

# Diversity-aware scheduler (injects dumb steps when population collapses):
python main.py --strategy diversity

# Custom smart ratio:
python main.py --smart-ratio 0.5
```

---

## Key Research Questions Implemented Here

| Q | Question | Where |
|---|----------|-------|
| Q1 | Does failure-informed mutation improve sample efficiency? | Compare `--smart-only` vs `--dumb-only` results |
| Q2 | When does the smart step help most? | Analyse `generation_logs` in JSON results |
| Q3 | How often to inject dumb steps? | Try `--strategy diversity` and vary `diversity_thresh` in `scheduler.py` |
| Q5 | Which failure signals matter most? | Modify `failure_evidence()` in `benchmark.py` |

---

## Benchmark Task

**Sorting algorithm optimization** — chosen because:
- ✅ Deterministic: same inputs every run → fully reproducible
- ✅ Gradable: correctness (70 pts) + speed vs Python built-in (30 pts)
- ✅ Analogous to AlphaEvolve's matrix multiplication task
- ✅ Easy to verify: `sort_array([3,1,2]) == [1,2,3]`

The seed program is a correct but slow **selection sort** (O(n²)).  
The search should discover faster algorithms (merge sort, quicksort, etc.).

---

## Results Format

Each run saves a JSON file in `results/run_<timestamp>.json`:
```json
{
  "best_score": 94.3,
  "total_evaluations": 87,
  "reached_target": false,
  "best_code": "def sort_array(arr): ...",
  "generations": [
    { "generation": 1, "best_score": 42.1, "step_type": "smart", ... },
    ...
  ]
}
```
Use this to plot sample efficiency curves (score vs evaluator calls).

---

## Next Steps

- [ ] Add more benchmark tasks (matrix operations, string problems)
- [ ] Implement multiple evaluator types (correctness-only, speed-only, combined)
- [ ] Add AutoResearch as a baseline in the ablation study
- [ ] Scale up: run 3+ seeds per configuration for statistical significance
