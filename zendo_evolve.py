"""
evolve.py
---------
Failure-informed evolutionary search for Zendo rule induction.

Three configs: dumb-only, smart-only, mixed (70% smart).
Baseline: openevolve-style (LLM sees examples but not which ones it got wrong).

Usage:
    python3 evolve.py --variant z1 --config smart --seed 42
    python3 evolve.py --variant z1 --config dumb  --seed 42
    python3 evolve.py --variant z1 --config mixed --seed 42
    python3 evolve.py --variant z1 --config openevolve --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from mat_framework.world import (World, sample_positives_for,
                                  sample_negatives_for, world_to_nl)
from mat_framework.rule import Rule, evaluate, rules_equivalent
from mat_framework.targets import phi as phi_of_variant, target_rule

# ── LLM client (Groq via OpenAI-compatible API) ──────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # fallback for local experiments — set this env var before running
    raise EnvironmentError("Set GROQ_API_KEY before running evolve.py")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")

def call_llm(prompt: str, temperature: float = 0.4) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()

# ── Rule helpers ──────────────────────────────────────────────────────────────

SEED_RULE = Rule.from_dict({
    "clauses": [{"conditions": [
        {"pred": "piece", "args": ["W", "P"]}
    ]}]
})  # trivially accepts every world

RULE_FORMAT_HINT = """
A rule is a JSON object with this schema:
{
  "clauses": [
    {
      "conditions": [
        {"pred": "piece",    "args": ["W", "P"]},
        {"pred": "blue",     "args": ["P"]},
        {"pred": "contact",  "args": ["P", "Q"]},
        {"pred": "red",      "args": ["Q"]}
      ]
    }
  ]
}

Available predicates:
  piece(W, P)        — P is a piece in world W  [always include this first]
  contact(P, Q)      — P and Q are touching
  same_coord1(P, Q)  — P and Q share the same x-coordinate
  red(P) / blue(P) / green(P)         — colour
  small(P) / medium(P) / large(P)     — size
  upright(P) / lhs(P) / rhs(P)        — orientation

Rules are OR over clauses; each clause is AND over conditions.
Variables: W = world, P/Q/R = pieces (uppercase strings).
Every clause must include piece(W, P) for each distinct piece variable used.
Return ONLY the JSON — no explanation.
"""

def rule_to_str(rule: Rule) -> str:
    return json.dumps(rule.to_dict(), indent=2)

def parse_rule_from_llm(text: str) -> Rule | None:
    # Extract first JSON block from LLM output
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(text[start:end])
        return Rule.from_dict(d)
    except Exception:
        return None

# ── Scoring ───────────────────────────────────────────────────────────────────

def _clause_accepts(clause_dict: dict, world: "World") -> bool:
    """Check if a single clause accepts the world."""
    try:
        single = Rule.from_dict({"clauses": [clause_dict]})
        return evaluate(single, world)
    except Exception:
        return False


def score_rule(rule: Rule, positives: list[World], negatives: list[World]) -> dict:
    tp = sum(1 for w in positives if evaluate(rule, w))
    tn = sum(1 for w in negatives if not evaluate(rule, w))
    fp = len(negatives) - tn
    fn = len(positives) - tp
    total = len(positives) + len(negatives)
    accuracy = (tp + tn) / total
    fp_worlds = [w for w in negatives if evaluate(rule, w)]
    fn_worlds = [w for w in positives if not evaluate(rule, w)]

    # Per-clause attribution: for each FP, which clause(s) accepted it?
    clauses = rule.to_dict()["clauses"]
    fp_attribution = []
    for w in fp_worlds:
        guilty = [i for i, c in enumerate(clauses) if _clause_accepts(c, w)]
        fp_attribution.append((w, guilty))

    # For each FN, which clause (if any) is "closest" (accepts it if we relax one condition)?
    fn_attribution = []
    for w in fn_worlds:
        fn_attribution.append((w, []))  # simpler: just show the world for now

    return {
        "accuracy": accuracy,
        "score_pct": round(accuracy * 100, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "false_positives": fp_worlds,
        "false_negatives": fn_worlds,
        "fp_attribution": fp_attribution,
        "fn_attribution": fn_attribution,
    }

# ── Mutation operators ────────────────────────────────────────────────────────

UNARY_PREDS = ["red","blue","green","small","medium","large","upright","lhs","rhs"]
BINARY_PREDS = ["contact", "same_coord1"]
COLORS = ["red", "blue", "green"]
SIZES  = ["small", "medium", "large"]
ORIENTS = ["upright", "lhs", "rhs"]
VALUE_FAMILIES = [COLORS, SIZES, ORIENTS]


def _random_clause(rng: random.Random) -> dict:
    """Generate a random meaningful clause with 2-5 conditions."""
    conds = [{"pred": "piece", "args": ["W", "P"]}]
    # Optionally add a unary attribute on P
    if rng.random() < 0.7:
        conds.append({"pred": rng.choice(COLORS),   "args": ["P"]})
    if rng.random() < 0.4:
        conds.append({"pred": rng.choice(SIZES),    "args": ["P"]})
    if rng.random() < 0.3:
        conds.append({"pred": rng.choice(ORIENTS),  "args": ["P"]})
    # Optionally add a binary relation to a second piece Q
    if rng.random() < 0.5:
        rel = rng.choice(BINARY_PREDS)
        conds.append({"pred": "piece",   "args": ["W", "Q"]})
        conds.append({"pred": rel,       "args": ["P", "Q"]})
        if rng.random() < 0.6:
            conds.append({"pred": rng.choice(COLORS),  "args": ["Q"]})
        if rng.random() < 0.3:
            conds.append({"pred": rng.choice(SIZES),   "args": ["Q"]})
    return {"conditions": conds}


def dumb_mutate(rule: Rule, rng: random.Random) -> Rule:
    """Random mutation: replace a clause or swap a predicate."""
    d = rule.to_dict()
    clauses = d["clauses"]
    if not clauses:
        return rule

    # Weight ops: replace_clause most common to escape trivial seed
    op = rng.choices(
        ["replace_clause", "add_clause", "swap_pred", "drop_condition"],
        weights=[4, 2, 3, 1],
        k=1
    )[0]

    clause_idx = rng.randrange(len(clauses))
    clause = clauses[clause_idx]
    conds = clause["conditions"]

    if op == "replace_clause":
        # Replace the selected clause with a freshly generated one
        clauses[clause_idx] = _random_clause(rng)

    elif op == "add_clause":
        d["clauses"].append(_random_clause(rng))

    elif op == "swap_pred":
        # Swap one unary predicate for another
        unary = [i for i, c in enumerate(conds) if c["pred"] in UNARY_PREDS]
        if unary:
            i = rng.choice(unary)
            family = next((f for f in VALUE_FAMILIES if conds[i]["pred"] in f), COLORS)
            alternatives = [p for p in family if p != conds[i]["pred"]] or UNARY_PREDS
            conds[i] = {"pred": rng.choice(alternatives), "args": conds[i]["args"]}

    elif op == "drop_condition":
        # Drop a non-piece condition (makes rule less restrictive)
        droppable = [i for i, c in enumerate(conds) if c["pred"] not in ("piece",)]
        if droppable:
            conds.pop(rng.choice(droppable))

    # Allow score to temporarily decrease to escape local minima
    # (use a wider acceptance window — handled in main loop)
    try:
        return Rule.from_dict(d)
    except Exception:
        return rule

# ── LLM prompts ──────────────────────────────────────────────────────────────

def smart_prompt_simple(current_rule: Rule, score_info: dict,
                        target_n_clauses: int = 1) -> str:
    """Simple prompt for exploration phase — show worlds, ask for improvement."""
    fp = score_info["false_positives"][:3]
    fn = score_info["false_negatives"][:3]
    fp_text = "\n".join(world_to_nl(w) for w in fp) if fp else "  (none)"
    fn_text = "\n".join(world_to_nl(w) for w in fn) if fn else "  (none)"
    return f"""You are finding a hidden rule for the Zendo game (exactly {target_n_clauses} clause(s)).
A world is POSITIVE (rule holds) or NEGATIVE (rule does not hold).

Current rule:
{rule_to_str(current_rule)}

Accuracy: {score_info['score_pct']}%  (fp={score_info['fp']} fn={score_info['fn']})

False positives (rule accepted these but they are NEGATIVE):
{fp_text}

False negatives (rule rejected these but they are POSITIVE):
{fn_text}

Propose a BETTER rule that fixes these errors.

{RULE_FORMAT_HINT}"""


def smart_prompt(current_rule: Rule, score_info: dict,
                 positives: list[World], negatives: list[World],
                 target_n_clauses: int = 1) -> str:
    clauses = current_rule.to_dict()["clauses"]
    n_clauses = len(clauses)

    # Build per-clause FP breakdown
    fp_lines = []
    for w, guilty_clauses in score_info["fp_attribution"][:4]:
        clause_labels = ", ".join(f"clause {i+1}" for i in guilty_clauses) if guilty_clauses else "no clause identified"
        fp_lines.append(f"  {world_to_nl(w)}\n  → Accepted by: {clause_labels} (TOO BROAD)")

    fn_lines = []
    for w, _ in score_info["fn_attribution"][:4]:
        # Show which clauses reject it and why
        rejected_by = []
        for i, c in enumerate(clauses):
            if not _clause_accepts(c, w):
                rejected_by.append(f"clause {i+1}")
        fn_lines.append(f"  {world_to_nl(w)}\n  → Rejected by all clauses ({', '.join(rejected_by)}) — rule is TOO SPECIFIC")

    fp_text = "\n".join(fp_lines) if fp_lines else "  (none)"
    fn_text = "\n".join(fn_lines) if fn_lines else "  (none)"

    # Clause-level diagnosis summary with contrast examples
    clause_diag = []
    all_guilty = {}
    for w, guilty in score_info["fp_attribution"]:
        for i in guilty:
            all_guilty[i] = all_guilty.get(i, 0) + 1

    # For each guilty clause, find worlds it CORRECTLY accepts (true positives)
    clause_tp_worlds = {}
    for i in all_guilty:
        tp_for_clause = [w for w in positives if _clause_accepts(clauses[i], w)]
        clause_tp_worlds[i] = tp_for_clause[:3]

    for i, count in sorted(all_guilty.items()):
        clause_diag.append(f"\n  Clause {i+1} causes {count} false positive(s) — needs more constraints")
        tp_examples = clause_tp_worlds.get(i, [])
        if tp_examples:
            tp_text = "; ".join(world_to_nl(w) for w in tp_examples)
            clause_diag.append(f"  Clause {i+1} CORRECTLY accepts (positive worlds, keep these): {tp_text}")
    clause_summary = "\n".join(clause_diag) if clause_diag else "  (no per-clause diagnosis)"

    return f"""You are finding a hidden rule for the Zendo game.
A world is POSITIVE (rule holds) or NEGATIVE (rule does not hold).

Current rule ({n_clauses} clause(s), OR-connected):
{rule_to_str(current_rule)}

Accuracy: {score_info['score_pct']}%  |  false_positives={score_info['fp']}  false_negatives={score_info['fn']}

CLAUSE DIAGNOSIS (compare TRUE POSITIVES vs FALSE POSITIVES to find what condition to add):
{clause_summary}

ALL FALSE POSITIVES (your rule accepted these NEGATIVE worlds):
{fp_text}

FALSE NEGATIVES (your rule rejected these POSITIVE worlds — rule is too strict):
{fn_text}

INSTRUCTIONS:
- Study the FALSE POSITIVES vs TRUE POSITIVES for each guilty clause.
- Find a condition that the TRUE POSITIVES share but the FALSE POSITIVES lack.
- Add that condition to tighten the clause.
- Do NOT remove conditions that help with true positives.
- The hidden rule has exactly {target_n_clauses} clause(s) total.

{RULE_FORMAT_HINT}"""

def openevolve_prompt(positives: list[World], negatives: list[World],
                      current_rule: Rule) -> str:
    pos_text = "\n".join(world_to_nl(w) for w in positives[:8])
    neg_text = "\n".join(world_to_nl(w) for w in negatives[:8])
    return f"""You are finding a hidden rule for the Zendo game.

POSITIVE worlds (rule holds):
{pos_text}

NEGATIVE worlds (rule does not hold):
{neg_text}

Current rule (propose something better):
{rule_to_str(current_rule)}

{RULE_FORMAT_HINT}"""

# ── Main evolutionary loop ────────────────────────────────────────────────────

def run(variant: str, config: str, seed: int, max_evals: int = 80) -> dict:
    rng = random.Random(seed)
    phi = phi_of_variant(variant)
    target = target_rule(variant)
    target_n_clauses = len(target.clauses)

    # Fixed dataset: 20 positives + 20 negatives
    positives = sample_positives_for(phi, 20, rng, prefix=f"{variant}_pos_s{seed}")
    negatives = sample_negatives_for(phi, 20, rng, prefix=f"{variant}_neg_s{seed}")

    current_rule = SEED_RULE
    best_rule = SEED_RULE
    best_score_seen = 0.0
    start_time = time.time()
    evals = 0
    history = []

    print(f"\n[{variant.upper()} | {config} | seed {seed}]")
    print(f"  Dataset: {len(positives)} positives, {len(negatives)} negatives")

    for gen in range(max_evals):
        evals += 1
        score_info = score_rule(current_rule, positives, negatives)
        exact = rules_equivalent(current_rule, target)

        history.append({
            "gen": gen,
            "score_pct": score_info["score_pct"],
            "exact_match": exact,
            "evals": evals,
        })

        print(f"  Gen {gen:3d} | score={score_info['score_pct']:5.1f}% | "
              f"fp={score_info['fp']} fn={score_info['fn']} | exact={exact}")

        # Primary: accuracy ≥ 95% (matches code-benchmark threshold)
        solved_accuracy = score_info["score_pct"] >= 95.0
        if solved_accuracy or exact:
            elapsed = time.time() - start_time
            label = "EXACT MATCH" if exact else "95%+ ACCURACY"
            print(f"  ✓ SOLVED ({label}) at eval {evals} in {elapsed:.1f}s")
            return {
                "variant": variant, "config": config, "seed": seed,
                "solved": True, "exact_match": exact,
                "evals": evals, "elapsed": elapsed,
                "final_score": score_info["score_pct"],
                "history": history,
            }

        # Choose step type
        if config == "dumb":
            step = "dumb"
        elif config == "smart":
            step = "smart"
        elif config == "openevolve":
            step = "openevolve"
        else:  # mixed — adaptive: reduce random rate as accuracy rises
            # When score ≥80% with no false negatives, we only need to
            # tighten existing clauses. Random mutations would destroy that
            # structure. Switch to pure smart for refinement.
            fn_now = score_info["fn"]
            score_now = score_info["score_pct"]
            if score_now >= 80.0 and fn_now == 0:
                smart_prob = 1.0   # refinement phase: pure smart
            elif score_now >= 65.0:
                smart_prob = 0.85  # mostly smart once structure is emerging
            else:
                smart_prob = 0.7   # exploration phase: standard 70/30
            step = "smart" if rng.random() < smart_prob else "dumb"

        # Propose next rule
        candidate = None
        if step == "dumb":
            candidate = dumb_mutate(current_rule, rng)
        elif step == "smart":
            try:
                # Refinement phase: score >= 80% with no FNs.
                # Use per-clause attribution + contrast examples to tighten specific clauses.
                # Also sample k=3 proposals and take the best (beam step).
                # Exploration phase: simpler prompt, single proposal.
                refinement_phase = (score_info["score_pct"] >= 80.0
                                    and score_info["fn"] == 0)
                if refinement_phase:
                    prompt = smart_prompt(current_rule, score_info, positives, negatives,
                                          target_n_clauses=target_n_clauses)
                    # Best-of-3: take the highest-scoring LLM proposal
                    proposals = []
                    for _ in range(3):
                        try:
                            out = call_llm(prompt, temperature=0.6)
                            p = parse_rule_from_llm(out)
                            if p is not None:
                                ps = score_rule(p, positives, negatives)
                                proposals.append((ps["score_pct"], p))
                        except Exception:
                            pass
                    if proposals:
                        proposals.sort(key=lambda x: x[0], reverse=True)
                        candidate = proposals[0][1]
                    else:
                        candidate = dumb_mutate(current_rule, rng)
                else:
                    prompt = smart_prompt_simple(current_rule, score_info,
                                                 target_n_clauses=target_n_clauses)
                    llm_out = call_llm(prompt)
                    candidate = parse_rule_from_llm(llm_out)
            except Exception as e:
                print(f"    [LLM error: {e}] falling back to dumb")
                candidate = dumb_mutate(current_rule, rng)
        elif step == "openevolve":
            try:
                prompt = openevolve_prompt(positives, negatives, current_rule)
                llm_out = call_llm(prompt)
                candidate = parse_rule_from_llm(llm_out)
            except Exception as e:
                print(f"    [LLM error: {e}]")
                candidate = None

        if candidate is None:
            continue

        cand_score = score_rule(candidate, positives, negatives)
        delta = cand_score["score_pct"] - score_info["score_pct"]

        # Acceptance rules:
        #   smart-config or openevolve: always allow -10% to escape local optima
        #   mixed in refinement phase (score>=80%, fn=0, step=smart): allow -10% so the
        #     LLM can tighten a specific clause without losing the exploration window
        #   mixed in exploration phase and dumb-step: strict (delta>=0), prevents cascading
        #   dumb-config: 15% stochastic accept so it can escape plateau (only exploration)
        in_refinement = (config == "mixed"
                         and score_info["score_pct"] >= 80.0
                         and score_info["fn"] == 0
                         and step == "smart")
        if config == "smart" or step == "openevolve" or in_refinement:
            accept = delta >= -10.0
        elif config == "dumb":
            accept = delta >= 0 or rng.random() < 0.15
        else:
            accept = delta >= 0

        if accept:
            current_rule = candidate
            if cand_score["score_pct"] > best_score_seen:
                best_rule = candidate
                best_score_seen = cand_score["score_pct"]
        # Periodically reset to best seen to prevent cascading backslide
        elif gen > 0 and gen % 10 == 0 and score_info["score_pct"] < best_score_seen - 5:
            current_rule = best_rule

    elapsed = time.time() - start_time
    final = score_rule(best_rule, positives, negatives)
    exact_final = rules_equivalent(best_rule, target)
    print(f"  ✗ NOT SOLVED after {evals} evals ({elapsed:.1f}s), "
          f"best score={final['score_pct']}%")
    return {
        "variant": variant, "config": config, "seed": seed,
        "solved": False, "exact_match": exact_final,
        "evals": evals, "elapsed": elapsed,
        "final_score": final["score_pct"],
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="z1", choices=["z1","z2","z3","z2b","z2c"])
    parser.add_argument("--config",  default="smart",
                        choices=["dumb","smart","mixed","openevolve"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--max-evals", type=int, default=80)
    args = parser.parse_args()

    result = run(args.variant, args.config, args.seed, args.max_evals)
    print(json.dumps(result, indent=2))
