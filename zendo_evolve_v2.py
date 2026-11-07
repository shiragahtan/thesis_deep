"""
zendo_evolve_v2.py
------------------
Extended Zendo solver comparing 7 methods:

  1. random-only    — pure random mutation baseline (no LLM)
  2. llm-only       — pure LLM ablation: no random steps, no failure signal
  3. openevolve     — TRUE OpenEvolve baseline: 70% LLM (no failure signal) + 30% random
                      (same 70/30 structure as our method but blind to which examples are wrong)
  4. our-evolve     — 70% LLM WITH failure signal + 30% random  ← MAIN METHOD
  5. distance       — our-evolve + predicate-Hamming annealed random steps
  6. distance-emb   — our-evolve + embedding-cosine annealed random steps (needs Ollama)
  7. version-space  — our-evolve + oracle queries for maximally discriminating worlds

All LLM methods use the same model (qwen/qwen3.8-27b on Groq or qwen2.5:7b on Ollama).
Embedding distance (distance-emb) uses nomic-embed-text via Ollama.

New features over v1:
  - Oracle query interface: phi(world) for any sampled world
  - Distance-based mutation: predicate Hamming distance + annealing schedule
  - Version-space search: maintain hypothesis pool, find maximally discriminating world
  - Rule mutation granularity: target worst-performing clause by FP count

Usage:
    python3 zendo_evolve_v2.py --variant z2 --config our-evolve --seed 42
    python3 zendo_evolve_v2.py --variant z2 --config distance --seed 42
    python3 zendo_evolve_v2.py --variant z2 --config version-space --seed 42
    python3 zendo_evolve_v2.py --variant z1 --config all --seed 42  # run all 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

from mat_framework.world import (World, Piece, sample_positives_for,
                                  sample_negatives_for, sample_world,
                                  world_to_nl, _derive_contacts)
from mat_framework.rule import (Rule, evaluate, rules_equivalent,
                                 predicates_used, within_limits, consistent)
from mat_framework.targets import phi as phi_of_variant, target_rule

# ── LLM client ────────────────────────────────────────────────────────────────

# Backend selection: set LLM_BACKEND=ollama to use a local Ollama model (no token limits).
# Default: groq (requires GROQ_API_KEY).
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq").lower()

if LLM_BACKEND == "ollama":
    LLM_API_KEY    = "ollama"
    LLM_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL      = os.environ.get("LLM_MODEL", "qwen2.5:7b")
    EMBED_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    EMBED_MODEL    = os.environ.get("EMBED_MODEL", "nomic-embed-text")
else:
    LLM_API_KEY    = os.environ.get("GROQ_API_KEY", "")
    LLM_BASE_URL   = "https://api.groq.com/openai/v1"
    LLM_MODEL      = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
    # Groq doesn't support embeddings — fall back to Ollama for embedding distance
    EMBED_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    EMBED_MODEL    = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    if not LLM_API_KEY:
        raise EnvironmentError("Set GROQ_API_KEY before running (or set LLM_BACKEND=ollama)")


def call_llm(prompt: str, temperature: float = 0.4) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    # max_tokens=512: Groq free tier has 1000 OTPM limit; a rule JSON fits in ~200 tokens.
    # Keeping this well below 1000 avoids OTPM 429 errors.
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ── Domain constants ──────────────────────────────────────────────────────────

UNARY_PREDS  = ["red", "blue", "green", "small", "medium", "large",
                 "upright", "lhs", "rhs"]
BINARY_PREDS = ["contact", "same_coord1"]
COLORS   = ["red", "blue", "green"]
SIZES    = ["small", "medium", "large"]
ORIENTS  = ["upright", "lhs", "rhs"]
VALUE_FAMILIES = [COLORS, SIZES, ORIENTS]

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

SEED_RULE = Rule.from_dict({"clauses": [{"conditions": [
    {"pred": "piece", "args": ["W", "P"]}
]}]})


# ── Distance helpers ──────────────────────────────────────────────────────────

def predicate_multiset(rule: Rule) -> list[str]:
    """All predicate names in the rule (duplicates kept per occurrence)."""
    return [c.pred for cl in rule.clauses for c in cl.conditions]


def predicate_distance(r1: Rule, r2: Rule) -> int:
    """Hamming-style distance on predicate multisets (symmetric difference size)."""
    m1 = defaultdict(int)
    m2 = defaultdict(int)
    for p in predicate_multiset(r1):
        m1[p] += 1
    for p in predicate_multiset(r2):
        m2[p] += 1
    all_keys = set(m1) | set(m2)
    return sum(abs(m1[k] - m2[k]) for k in all_keys)


def semantic_distance(r1: Rule, r2: Rule, worlds: list[World]) -> float:
    """Fraction of worlds where r1 and r2 disagree."""
    if not worlds:
        return 0.0
    disagree = sum(1 for w in worlds if evaluate(r1, w) != evaluate(r2, w))
    return disagree / len(worlds)


def _rule_to_nl(rule: Rule) -> str:
    """Human-readable string for a rule, used as embedding input."""
    clauses = rule.to_dict()["clauses"]
    parts = []
    for i, cl in enumerate(clauses):
        preds = [f"{c['pred']}({','.join(c['args'])})" for c in cl["conditions"]]
        parts.append(f"clause{i+1}: " + " AND ".join(preds))
    return " OR ".join(parts)


_embed_cache: dict[str, list[float]] = {}

def _embed(text: str) -> list[float]:
    """Get embedding vector for text via Ollama (cached)."""
    if text in _embed_cache:
        return _embed_cache[text]
    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=EMBED_BASE_URL)
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        vec = resp.data[0].embedding
        _embed_cache[text] = vec
        return vec
    except Exception:
        return []


def embedding_distance(r1: Rule, r2: Rule) -> float:
    """Cosine distance between rule embeddings (0=identical, 1=orthogonal, 2=opposite).

    Uses Ollama's nomic-embed-text (or EMBED_MODEL) to embed the NL rule description.
    Falls back to predicate_distance if embedding fails.
    """
    v1 = _embed(_rule_to_nl(r1))
    v2 = _embed(_rule_to_nl(r2))
    if not v1 or not v2 or len(v1) != len(v2):
        return predicate_distance(r1, r2) / 10.0  # normalised fallback
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 1.0
    cosine_sim = dot / (n1 * n2)
    return 1.0 - cosine_sim  # distance: 0=identical, higher=more different


# ── Rule generation and mutation ──────────────────────────────────────────────

def _random_clause(rng: random.Random) -> dict:
    """Generate a random meaningful clause with 2-5 conditions."""
    conds = [{"pred": "piece", "args": ["W", "P"]}]
    if rng.random() < 0.7:
        conds.append({"pred": rng.choice(COLORS),   "args": ["P"]})
    if rng.random() < 0.4:
        conds.append({"pred": rng.choice(SIZES),    "args": ["P"]})
    if rng.random() < 0.3:
        conds.append({"pred": rng.choice(ORIENTS),  "args": ["P"]})
    if rng.random() < 0.5:
        rel = rng.choice(BINARY_PREDS)
        conds.append({"pred": "piece",  "args": ["W", "Q"]})
        conds.append({"pred": rel,      "args": ["P", "Q"]})
        if rng.random() < 0.6:
            conds.append({"pred": rng.choice(COLORS), "args": ["Q"]})
        if rng.random() < 0.3:
            conds.append({"pred": rng.choice(SIZES),  "args": ["Q"]})
    return {"conditions": conds}


def _clauses_distance(c1: dict, c2: dict) -> int:
    """Predicate Hamming between two clause dicts."""
    m1 = defaultdict(int)
    m2 = defaultdict(int)
    for cond in c1["conditions"]:
        m1[cond["pred"]] += 1
    for cond in c2["conditions"]:
        m2[cond["pred"]] += 1
    all_keys = set(m1) | set(m2)
    return sum(abs(m1[k] - m2[k]) for k in all_keys)


def _dumb_mutate_clause_bounded(clause: dict, rng: random.Random,
                                  max_distance: int) -> dict:
    """Mutate a single clause, keeping predicate-distance from original ≤ max_distance."""
    for _ in range(20):
        new_clause = _dumb_mutate_clause(clause, rng)
        if _clauses_distance(clause, new_clause) <= max_distance:
            return new_clause
    return _random_clause(rng)


def _dumb_mutate_clause(clause: dict, rng: random.Random) -> dict:
    """Single-clause mutation: swap pred, drop condition, add condition."""
    import copy
    clause = copy.deepcopy(clause)
    conds = clause["conditions"]

    op = rng.choices(["swap_pred", "drop_condition", "add_condition"],
                     weights=[4, 2, 2], k=1)[0]

    if op == "swap_pred":
        unary = [i for i, c in enumerate(conds) if c["pred"] in UNARY_PREDS]
        if unary:
            i = rng.choice(unary)
            family = next((f for f in VALUE_FAMILIES if conds[i]["pred"] in f), COLORS)
            alternatives = [p for p in family if p != conds[i]["pred"]] or UNARY_PREDS
            conds[i] = {"pred": rng.choice(alternatives), "args": conds[i]["args"]}
    elif op == "drop_condition":
        droppable = [i for i, c in enumerate(conds) if c["pred"] != "piece"]
        if droppable:
            conds.pop(rng.choice(droppable))
    elif op == "add_condition":
        vars_in_clause = {a for c in conds for a in c["args"]
                          if a and a[0].isupper() and a != "W"}
        if rng.random() < 0.5 and vars_in_clause:
            v = rng.choice(list(vars_in_clause))
            conds.append({"pred": rng.choice(UNARY_PREDS), "args": [v]})
        else:
            conds.append({"pred": rng.choice(UNARY_PREDS), "args": ["P"]})

    return clause


def dumb_mutate(rule: Rule, rng: random.Random,
                target_clause_idx: int | None = None) -> Rule:
    """Random mutation. If target_clause_idx given, mutate only that clause."""
    import copy
    d = rule.to_dict()
    clauses = d["clauses"]
    if not clauses:
        return rule

    if target_clause_idx is not None and 0 <= target_clause_idx < len(clauses):
        idx = target_clause_idx
    else:
        op = rng.choices(["mutate_clause", "replace_clause", "add_clause"],
                         weights=[3, 3, 1], k=1)[0]
        idx = rng.randrange(len(clauses))
        if op == "replace_clause":
            d["clauses"][idx] = _random_clause(rng)
            try:
                return Rule.from_dict(d)
            except Exception:
                return rule
        elif op == "add_clause":
            d["clauses"].append(_random_clause(rng))
            try:
                return Rule.from_dict(d)
            except Exception:
                return rule

    clauses[idx] = _dumb_mutate_clause(clauses[idx], rng)
    try:
        return Rule.from_dict(d)
    except Exception:
        return rule


def distance_mutate(rule: Rule, rng: random.Random,
                    max_step: float, target_clause_idx: int | None = None) -> Rule:
    """Distance-aware mutation: predicate Hamming distance from original ≤ max_step.

    At large max_step (exploration phase), allows big jumps (replace_clause).
    At small max_step (refinement phase), restricts to small edits.
    """
    import copy
    d = copy.deepcopy(rule.to_dict())
    clauses = d["clauses"]
    if not clauses:
        return rule

    if target_clause_idx is not None:
        idx = target_clause_idx
    else:
        idx = rng.randrange(len(clauses))

    # Large step: full replacement allowed
    if max_step >= 6:
        d["clauses"][idx] = _random_clause(rng)
    else:
        d["clauses"][idx] = _dumb_mutate_clause_bounded(
            clauses[idx], rng, max_distance=int(max_step))

    try:
        return Rule.from_dict(d)
    except Exception:
        return rule


def anneal_step(gen: int, max_evals: int, initial: float = 10.0, final: float = 1.0) -> float:
    """Exponential annealing: max_step decreases from initial to final over eval budget."""
    t = gen / max(max_evals - 1, 1)
    return final + (initial - final) * (1 - t) ** 2


# ── Scoring ───────────────────────────────────────────────────────────────────

def _clause_accepts(clause_dict: dict, world: World) -> bool:
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

    clauses = rule.to_dict()["clauses"]
    fp_attribution = []
    for w in fp_worlds:
        guilty = [i for i, c in enumerate(clauses) if _clause_accepts(c, w)]
        fp_attribution.append((w, guilty))
    fn_attribution = [(w, []) for w in fn_worlds]

    # Per-clause: how many positives each clause covers (TP per clause)
    clause_coverage = []
    for i, c in enumerate(clauses):
        covered = sum(1 for w in positives if _clause_accepts(c, w))
        clause_coverage.append(covered)

    return {
        "accuracy": accuracy,
        "score_pct": round(accuracy * 100, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "false_positives": fp_worlds,
        "false_negatives": fn_worlds,
        "fp_attribution": fp_attribution,
        "fn_attribution": fn_attribution,
        "clause_coverage": clause_coverage,  # positives covered by each clause
    }


# ── LLM prompts ──────────────────────────────────────────────────────────────

def _prompt_simple(rule: Rule, score_info: dict, n_clauses: int) -> str:
    fp = score_info["false_positives"][:3]
    fn = score_info["false_negatives"][:3]
    fp_text = "\n".join(f"  {world_to_nl(w)}" for w in fp) if fp else "  (none)"
    fn_text = "\n".join(f"  {world_to_nl(w)}" for w in fn) if fn else "  (none)"

    coverage = score_info.get("clause_coverage", [])
    if n_clauses > 1 and coverage:
        cov_lines = "\n".join(
            f"  Clause {i+1}: covers {c}/{len(score_info['false_negatives']) + score_info['tp']} positives"
            for i, c in enumerate(coverage)
        )
        multi_hint = f"""
PER-CLAUSE COVERAGE (how many positives each clause fires on):
{cov_lines}
Tip: a {n_clauses}-clause OR rule succeeds if ANY clause matches.
Each clause should independently cover a DIFFERENT subset of positives.
"""
    else:
        multi_hint = ""

    return f"""You are finding a hidden rule for the Zendo game (exactly {n_clauses} clause(s)).
A world is POSITIVE (rule holds) or NEGATIVE (rule does not hold).
The rule fires if ANY of its clauses matches (OR logic).

Current rule:
{json.dumps(rule.to_dict(), indent=2)}

Accuracy: {score_info['score_pct']}%  (fp={score_info['fp']} fn={score_info['fn']})
{multi_hint}
False positives (rule accepted these but they are NEGATIVE):
{fp_text}

False negatives (rule rejected these but they are POSITIVE):
{fn_text}

Propose a BETTER rule that fixes these errors.

{RULE_FORMAT_HINT}"""


def _prompt_attributed(rule: Rule, score_info: dict,
                       positives: list[World], negatives: list[World],
                       n_clauses: int) -> str:
    clauses = rule.to_dict()["clauses"]
    fp_lines = []
    for w, guilty in score_info["fp_attribution"][:4]:
        label = ", ".join(f"clause {i+1}" for i in guilty) if guilty else "unknown"
        fp_lines.append(f"  {world_to_nl(w)}\n  → Accepted by: {label} (TOO BROAD)")
    fn_lines = []
    for w, _ in score_info["fn_attribution"][:4]:
        rejected = [f"clause {i+1}" for i, c in enumerate(clauses)
                    if not _clause_accepts(c, w)]
        fn_lines.append(f"  {world_to_nl(w)}\n  → Rejected by: {', '.join(rejected)}")
    fp_text = "\n".join(fp_lines) if fp_lines else "  (none)"
    fn_text = "\n".join(fn_lines) if fn_lines else "  (none)"

    # Per-clause: show TP examples alongside FP examples
    guilty_counts: dict[int, int] = {}
    for _, guilty in score_info["fp_attribution"]:
        for i in guilty:
            guilty_counts[i] = guilty_counts.get(i, 0) + 1
    diag_lines = []
    for i, count in sorted(guilty_counts.items()):
        tp_ex = [w for w in positives if _clause_accepts(clauses[i], w)][:3]
        diag_lines.append(
            f"\n  Clause {i+1} causes {count} false positive(s)"
        )
        if tp_ex:
            tp_text = "; ".join(world_to_nl(w) for w in tp_ex)
            diag_lines.append(f"  Clause {i+1} CORRECTLY accepts (keep): {tp_text}")
    clause_summary = "\n".join(diag_lines) if diag_lines else "  (no per-clause diagnosis)"

    return f"""You are finding a hidden rule for the Zendo game.

Current rule ({len(clauses)} clause(s), OR-connected):
{json.dumps(rule.to_dict(), indent=2)}

Accuracy: {score_info['score_pct']}%  |  fp={score_info['fp']}  fn={score_info['fn']}

CLAUSE DIAGNOSIS:
{clause_summary}

FALSE POSITIVES (rule accepted these NEGATIVE worlds):
{fp_text}

FALSE NEGATIVES (rule rejected these POSITIVE worlds):
{fn_text}

INSTRUCTIONS:
- For each guilty clause: add a condition present in the CORRECT worlds but absent in FP worlds.
- Do NOT remove conditions that help with true positives.
- The rule has exactly {n_clauses} clause(s) total.

{RULE_FORMAT_HINT}"""


def _prompt_openevolve(positives: list[World], negatives: list[World],
                       rule: Rule) -> str:
    pos_text = "\n".join(world_to_nl(w) for w in positives[:5])
    neg_text = "\n".join(world_to_nl(w) for w in negatives[:5])
    return f"""You are finding a hidden rule for the Zendo game.

POSITIVE worlds (rule holds):
{pos_text}

NEGATIVE worlds (rule does not hold):
{neg_text}

Current rule (propose something better):
{json.dumps(rule.to_dict(), indent=2)}

{RULE_FORMAT_HINT}"""


def _prompt_version_space_query(h_pool: list[Rule], n_clauses: int,
                                 existing_evidence: list[tuple[World, bool]]) -> str:
    """Ask the LLM to propose a discriminating world to distinguish between hypotheses."""
    sample_rules = h_pool[:4]
    rules_text = "\n\n".join(
        f"Hypothesis {i+1}:\n{json.dumps(r.to_dict(), indent=2)}"
        for i, r in enumerate(sample_rules)
    )
    evidence_text = "\n".join(
        f"  World {w.id}: {'POSITIVE' if lbl else 'NEGATIVE'}"
        for w, lbl in existing_evidence[-5:]
    ) or "  (none yet)"

    return f"""You are helping to identify the correct hidden rule in the Zendo game.

There are {len(h_pool)} candidate rules. Here are some of them:

{rules_text}

Recent oracle evidence:
{evidence_text}

Your task: design a world that BEST DISTINGUISHES between these hypotheses.
The ideal world should be POSITIVE under some hypotheses and NEGATIVE under others,
splitting them as evenly as possible.

Describe the world as a JSON object with this schema:
{{
  "pieces": [
    {{"color": "red", "size": 5, "orient": "upright", "x": 3, "y": 4}},
    {{"color": "blue", "size": 2, "orient": "lhs",    "x": 3, "y": 5}}
  ]
}}

Colors: red, blue, green
Orientations: upright, lhs, rhs
Size: 0-10 (small≤3, medium 4-6, large≥7)
x, y: 0-10

Return ONLY the JSON world description — no explanation.
"""


def parse_rule_from_llm(text: str) -> Rule | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(text[start:end])
        return Rule.from_dict(d)
    except Exception:
        return None


def parse_world_from_llm(text: str, world_id: str) -> World | None:
    """Parse a world description from LLM output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(text[start:end])
        pieces_data = d.get("pieces", [])
        pieces = []
        for i, pd in enumerate(pieces_data):
            pieces.append(Piece(
                id=f"{world_id}_p{i}",
                x=int(pd.get("x", 5)),
                y=int(pd.get("y", 5)),
                size=int(pd.get("size", 5)),
                color=pd.get("color", "red"),
                orient=pd.get("orient", "upright"),
            ))
        return World(id=world_id, pieces=pieces, contacts=_derive_contacts(pieces))
    except Exception:
        return None


# ── Version-space helpers ─────────────────────────────────────────────────────

def _generate_hypothesis_pool(n: int, rng: random.Random,
                               evidence: list[tuple[World, bool]],
                               n_clauses: int) -> list[Rule]:
    """Generate a pool of random rules consistent with labeled evidence."""
    pool = []
    attempts = 0
    while len(pool) < n and attempts < n * 50:
        attempts += 1
        # Build a rule with the right number of clauses
        clauses = [_random_clause(rng) for _ in range(n_clauses)]
        try:
            rule = Rule.from_dict({"clauses": clauses})
            if consistent(rule, evidence):
                pool.append(rule)
        except Exception:
            pass
    return pool


def _best_discriminating_world(h_pool: list[Rule],
                                 candidate_worlds: list[World]) -> World | None:
    """Find the world W in candidates that best splits the hypothesis pool.

    Optimal split: maximize min(|pos_subset|, |neg_subset|).
    """
    if not h_pool or not candidate_worlds:
        return None
    best_world = None
    best_split = -1
    for w in candidate_worlds:
        pos_count = sum(1 for r in h_pool if evaluate(r, w))
        neg_count = len(h_pool) - pos_count
        split_quality = min(pos_count, neg_count)
        if split_quality > best_split:
            best_split = split_quality
            best_world = w
    return best_world


def _eliminate_inconsistent(h_pool: list[Rule],
                              world: World, oracle_label: bool) -> list[Rule]:
    """Remove hypotheses inconsistent with the oracle's answer for world."""
    return [r for r in h_pool if evaluate(r, world) == oracle_label]


# ── Main evolutionary loop ────────────────────────────────────────────────────

def run(variant: str, config: str, seed: int, max_evals: int = 80,
        base_smart_prob: float = 0.7) -> dict:
    """Run one evolutionary search trial.

    base_smart_prob: fraction of steps that use the LLM at the exploration phase
        (score < 65%). Default 0.7 = 70/30 split. Set to 0.5 for 50/50, 0.9 for 90/10.
        Adaptive scaling still applies on top of this:
            score >= 80% and fn==0 → 1.0 (pure LLM refinement)
            score >= 65%            → base + 0.5*(1-base)
            else                    → base_smart_prob
    """
    rng = random.Random(seed)
    phi = phi_of_variant(variant)
    target = target_rule(variant)
    n_clauses = len(target.clauses)

    # Fixed labeled dataset
    positives = sample_positives_for(phi, 20, rng, prefix=f"{variant}_pos_s{seed}")
    negatives = sample_negatives_for(phi, 20, rng, prefix=f"{variant}_neg_s{seed}")

    # Oracle evidence accumulator (for version-space)
    oracle_evidence: list[tuple[World, bool]] = (
        [(w, True) for w in positives] + [(w, False) for w in negatives]
    )

    # Unlabeled world pool for oracle queries and version-space discriminators
    # (sample a large set once; oracle labels them on demand)
    query_pool = [sample_world(f"{variant}_q_s{seed}_{i}", rng)
                  for i in range(200)]

    current_rule = SEED_RULE
    best_rule = SEED_RULE
    best_score_seen = 0.0
    start_time = time.time()
    evals = 0
    history = []
    oracle_queries = 0

    # Version-space: hypothesis pool (consistent with current evidence)
    h_pool: list[Rule] = []
    if config == "version-space":
        h_pool = _generate_hypothesis_pool(50, rng, oracle_evidence, n_clauses)
        print(f"  Version-space: initial pool = {len(h_pool)} hypotheses")

    print(f"\n[{variant.upper()} | {config} | seed {seed}]")
    print(f"  Dataset: {len(positives)} positives, {len(negatives)} negatives")

    stall_count = 0  # consecutive gens with no improvement
    last_improvement_score = 0.0

    for gen in range(max_evals):
        evals += 1
        score_info = score_rule(current_rule, positives, negatives)
        exact = rules_equivalent(current_rule, target)

        # Dead-rule detection: if ALL clauses cover 0 positives, restart from best seen.
        # This avoids wasting budget on a completely non-firing rule.
        if (n_clauses > 1
                and all(c == 0 for c in score_info.get("clause_coverage", [1]))
                and best_score_seen > 0):
            current_rule = best_rule
            score_info = score_rule(current_rule, positives, negatives)

        # Stall detection: if score hasn't improved in last 8 gens, reset to best.
        if score_info["score_pct"] > last_improvement_score:
            last_improvement_score = score_info["score_pct"]
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= 8 and best_score_seen > score_info["score_pct"]:
            current_rule = best_rule
            score_info = score_rule(current_rule, positives, negatives)  # keep in sync
            stall_count = 0

        history.append({
            "gen": gen,
            "score_pct": score_info["score_pct"],
            "exact_match": exact,
            "evals": evals,
            "oracle_queries": oracle_queries,
        })

        print(f"  Gen {gen:3d} | score={score_info['score_pct']:5.1f}% | "
              f"fp={score_info['fp']} fn={score_info['fn']} | exact={exact}"
              + (f" | pool={len(h_pool)}" if config == "version-space" else ""))

        if score_info["score_pct"] >= 95.0 or exact:
            elapsed = time.time() - start_time
            label = "EXACT" if exact else "95%+"
            print(f"  SOLVED ({label}) at eval {evals} in {elapsed:.1f}s "
                  f"[oracle_queries={oracle_queries}]")
            return {
                "variant": variant, "config": config, "seed": seed,
                "solved": True, "exact_match": exact,
                "evals": evals, "elapsed": elapsed,
                "final_score": score_info["score_pct"],
                "oracle_queries": oracle_queries,
                "history": history,
            }

        # ── Step selection per config ──────────────────────────────────────────

        if config == "random-only":
            candidate = dumb_mutate(current_rule, rng)

        elif config == "llm-only":
            # Pure LLM ablation — no random steps, no failure signal.
            # Not the same as OpenEvolve (which uses random steps too).
            try:
                prompt = _prompt_openevolve(positives, negatives, current_rule)
                out = call_llm(prompt)
                candidate = parse_rule_from_llm(out) or dumb_mutate(current_rule, rng)
            except Exception as e:
                print(f"    [LLM error: {e}]")
                candidate = dumb_mutate(current_rule, rng)

        elif config == "openevolve":
            # True OpenEvolve equivalent: 70% LLM (no failure signal) + 30% random.
            # Same 70/30 split as our-evolve but LLM sees examples without knowing
            # which are wrong — matching AlphaEvolve's blind-to-failure architecture.
            score_now = score_info["score_pct"]
            fn_now = score_info["fn"]
            if score_now >= 80.0 and fn_now == 0:
                smart_prob = 1.0
            elif score_now >= 65.0:
                smart_prob = base_smart_prob + 0.5 * (1 - base_smart_prob)
            else:
                smart_prob = base_smart_prob
            use_llm = rng.random() < smart_prob
            if not use_llm:
                candidate = dumb_mutate(current_rule, rng)
            else:
                try:
                    prompt = _prompt_openevolve(positives, negatives, current_rule)
                    out = call_llm(prompt)
                    candidate = parse_rule_from_llm(out) or dumb_mutate(current_rule, rng)
                except Exception as e:
                    print(f"    [LLM error: {e}]")
                    candidate = dumb_mutate(current_rule, rng)

        elif config in ("failure-informed", "our-evolve"):
            # failure-informed: pure LLM with failure signal (smart-only ablation)
            # our-evolve:       base_smart_prob% LLM with failure signal + rest random
            #   adaptive: when score>=80% and fn=0, switch to 100% LLM for refinement
            if config == "our-evolve":
                fn_now = score_info["fn"]
                score_now = score_info["score_pct"]
                if score_now >= 80.0 and fn_now == 0:
                    smart_prob = 1.0
                elif score_now >= 65.0:
                    smart_prob = base_smart_prob + 0.5 * (1 - base_smart_prob)
                else:
                    smart_prob = base_smart_prob
                use_llm = rng.random() < smart_prob
            else:
                use_llm = True  # failure-informed always uses LLM

            if not use_llm:
                candidate = dumb_mutate(current_rule, rng)
            else:
                try:
                    refinement = (score_info["score_pct"] >= 80.0 and score_info["fn"] == 0)
                    if refinement:
                        prompt = _prompt_attributed(current_rule, score_info,
                                                    positives, negatives, n_clauses)
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
                        prompt = _prompt_simple(current_rule, score_info, n_clauses)
                        out = call_llm(prompt)
                        candidate = parse_rule_from_llm(out) or dumb_mutate(current_rule, rng)
                except Exception as e:
                    print(f"    [LLM error: {e}]")
                    candidate = dumb_mutate(current_rule, rng)

        elif config == "distance":
            # our-evolve + distance-annealed mutations.
            # Same 70/30 split as our-evolve, but random steps are distance-bounded.
            # Annealing: large steps early (exploration), small steps late (refinement).
            max_step = anneal_step(gen, max_evals, initial=10.0, final=1.0)

            # Identify which clause is most responsible for FPs
            guilty_counts: dict[int, int] = {}
            for _, guilty in score_info["fp_attribution"]:
                for i in guilty:
                    guilty_counts[i] = guilty_counts.get(i, 0) + 1
            worst_clause = (max(guilty_counts, key=guilty_counts.get)
                            if guilty_counts else None)

            # Same adaptive split as our-evolve, using base_smart_prob
            score_now = score_info["score_pct"]
            fn_now = score_info["fn"]
            if score_now >= 80.0 and fn_now == 0:
                smart_prob = 1.0
            elif score_now >= 65.0:
                smart_prob = base_smart_prob + 0.5 * (1 - base_smart_prob)
            else:
                smart_prob = base_smart_prob

            use_llm = rng.random() < smart_prob
            if not use_llm:
                # Distance-bounded random step (the key difference from our-evolve)
                candidate = distance_mutate(current_rule, rng, max_step, worst_clause)
            else:
                try:
                    refinement = (score_now >= 80.0 and fn_now == 0)
                    if refinement:
                        prompt = _prompt_attributed(current_rule, score_info,
                                                    positives, negatives, n_clauses)
                        out = call_llm(prompt, temperature=0.6)
                    else:
                        prompt = _prompt_simple(current_rule, score_info, n_clauses)
                        out = call_llm(prompt)
                    candidate = parse_rule_from_llm(out)
                    if candidate is None:
                        candidate = distance_mutate(current_rule, rng, max_step, worst_clause)
                except Exception as e:
                    print(f"    [LLM error: {e}]")
                    candidate = distance_mutate(current_rule, rng, max_step, worst_clause)

        elif config == "distance-emb":
            # our-evolve + embedding-space distance annealing.
            # Random steps are filtered by cosine distance in rule embedding space
            # (via nomic-embed-text). Requires Ollama running with nomic-embed-text.
            max_step_emb = anneal_step(gen, max_evals, initial=0.5, final=0.05)

            guilty_counts: dict[int, int] = {}
            for _, guilty in score_info["fp_attribution"]:
                for i in guilty:
                    guilty_counts[i] = guilty_counts.get(i, 0) + 1
            worst_clause = (max(guilty_counts, key=guilty_counts.get)
                            if guilty_counts else None)

            score_now = score_info["score_pct"]
            fn_now = score_info["fn"]
            if score_now >= 80.0 and fn_now == 0:
                smart_prob = 1.0
            elif score_now >= 65.0:
                smart_prob = base_smart_prob + 0.5 * (1 - base_smart_prob)
            else:
                smart_prob = base_smart_prob

            use_llm = rng.random() < smart_prob
            if not use_llm:
                # Generate candidates and pick the one closest to current rule
                # within the annealed embedding distance budget
                best_cand = None
                best_dist = float("inf")
                for _ in range(10):
                    cand = dumb_mutate(current_rule, rng, worst_clause)
                    d = embedding_distance(current_rule, cand)
                    if d <= max_step_emb and d < best_dist:
                        best_dist = d
                        best_cand = cand
                candidate = best_cand if best_cand is not None else dumb_mutate(current_rule, rng)
            else:
                try:
                    refinement = (score_now >= 80.0 and fn_now == 0)
                    if refinement:
                        prompt = _prompt_attributed(current_rule, score_info,
                                                    positives, negatives, n_clauses)
                        out = call_llm(prompt, temperature=0.6)
                    else:
                        prompt = _prompt_simple(current_rule, score_info, n_clauses)
                        out = call_llm(prompt)
                    candidate = parse_rule_from_llm(out)
                    if candidate is None:
                        candidate = dumb_mutate(current_rule, rng, worst_clause)
                except Exception as e:
                    print(f"    [LLM error: {e}]")
                    candidate = dumb_mutate(current_rule, rng, worst_clause)

        elif config == "version-space":
            # Every K steps, use version-space oracle query to prune hypotheses.
            # Between queries, use failure-informed LLM or distance mutation.

            VS_QUERY_INTERVAL = 5

            if gen % VS_QUERY_INTERVAL == 0 and len(h_pool) > 1:
                # Find best discriminating world using our algorithm
                disc_world = _best_discriminating_world(h_pool, query_pool)

                if disc_world is None:
                    # Fall back to LLM-generated discriminating world
                    try:
                        prompt = _prompt_version_space_query(
                            h_pool, n_clauses, oracle_evidence[-10:])
                        out = call_llm(prompt, temperature=0.5)
                        disc_world = parse_world_from_llm(out, f"vs_q{gen}")
                    except Exception as e:
                        print(f"    [VS query LLM error: {e}]")

                if disc_world is not None:
                    oracle_label = phi(disc_world)
                    oracle_queries += 1
                    oracle_evidence.append((disc_world, oracle_label))
                    old_size = len(h_pool)
                    h_pool = _eliminate_inconsistent(h_pool, disc_world, oracle_label)
                    print(f"    [VS] oracle({disc_world.id})={'POS' if oracle_label else 'NEG'} "
                          f"| pool: {old_size} → {len(h_pool)}")

                    # Replenish pool if too small
                    if len(h_pool) < 10:
                        new_hyps = _generate_hypothesis_pool(
                            30, rng, oracle_evidence, n_clauses)
                        h_pool.extend(new_hyps)
                        print(f"    [VS] replenished pool to {len(h_pool)}")

                    # Pick the best hypothesis in the pool as the new current rule
                    if h_pool:
                        ranked = sorted(
                            h_pool,
                            key=lambda r: score_rule(r, positives, negatives)["score_pct"],
                            reverse=True
                        )
                        candidate = ranked[0]
                    else:
                        candidate = dumb_mutate(current_rule, rng)
                else:
                    candidate = dumb_mutate(current_rule, rng)
            else:
                # Non-query step: failure-informed mutation
                try:
                    refinement = (score_info["score_pct"] >= 80.0 and score_info["fn"] == 0)
                    if refinement:
                        prompt = _prompt_attributed(current_rule, score_info,
                                                    positives, negatives, n_clauses)
                        proposals = []
                        for _ in range(2):
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
                        prompt = _prompt_simple(current_rule, score_info, n_clauses)
                        out = call_llm(prompt)
                        candidate = parse_rule_from_llm(out) or dumb_mutate(current_rule, rng)
                except Exception as e:
                    print(f"    [LLM error: {e}]")
                    candidate = dumb_mutate(current_rule, rng)
        else:
            raise ValueError(f"Unknown config: {config}")

        if candidate is None:
            continue

        cand_score = score_rule(candidate, positives, negatives)
        delta = cand_score["score_pct"] - score_info["score_pct"]

        # Acceptance: allow small drops to escape local optima, but clamp to avoid drift.
        # The allowed drop narrows as we approach a good solution (simulated annealing spirit).
        if config in ("random-only",):
            accept = delta >= 0 or rng.random() < 0.15
        else:
            # Allow at most a 5% drop (was 10% — too permissive for multi-clause rules).
            accept = delta >= -5.0

        if accept:
            current_rule = candidate
            if cand_score["score_pct"] > best_score_seen:
                best_rule = candidate
                best_score_seen = cand_score["score_pct"]

        # Reset to best seen if we've drifted more than 5% below it for 5+ gens.
        if gen > 0 and gen % 5 == 0 and score_info["score_pct"] < best_score_seen - 5:
            current_rule = best_rule

    elapsed = time.time() - start_time
    final = score_rule(best_rule, positives, negatives)
    exact_final = rules_equivalent(best_rule, target)
    print(f"  NOT SOLVED after {evals} evals ({elapsed:.1f}s), "
          f"best={final['score_pct']}% [oracle_queries={oracle_queries}]")
    return {
        "variant": variant, "config": config, "seed": seed,
        "solved": False, "exact_match": exact_final,
        "evals": evals, "elapsed": elapsed,
        "final_score": final["score_pct"],
        "oracle_queries": oracle_queries,
        "history": history,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

CONFIGS = ["random-only", "llm-only", "openevolve", "our-evolve",
           "distance", "distance-emb", "version-space"]
# llm-only       = pure LLM ablation (no random, no failure signal)
# openevolve     = true OpenEvolve: 70% LLM (no failure signal) + 30% random
# our-evolve     = 70% LLM (WITH failure signal) + 30% random  ← main method
# distance       = our-evolve + predicate-Hamming annealed random steps
# distance-emb   = our-evolve + embedding-cosine annealed random steps
# failure-informed = LLM-only ablation of our-evolve, still runnable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",    default="z1",
                        choices=["z1", "z2", "z3"])
    parser.add_argument("--config",    default="our-evolve",
                        choices=CONFIGS + ["all", "failure-informed", "all-extended"])
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--smart-prob", type=float, default=0.7,
                        help="Base LLM fraction for our-evolve/openevolve/distance "
                             "(0.5=50/50, 0.7=70/30, 0.9=90/10). "
                             "Default 0.7. Adaptive scaling applied on top.")
    args = parser.parse_args()

    if args.config == "all-extended":
        # Runs all configs including openevolve and distance-emb (needs Ollama)
        run_configs = CONFIGS
    elif args.config == "all":
        # Original 5-method comparison (no embedding distance, faster)
        run_configs = ["random-only", "llm-only", "openevolve", "our-evolve",
                       "distance", "version-space"]
    else:
        run_configs = None

    smart_prob = args.smart_prob

    if run_configs is not None:
        results = []
        for cfg in run_configs:
            r = run(args.variant, cfg, args.seed, args.max_evals, smart_prob)
            results.append(r)
        print("\n=== SUMMARY ===")
        print(f"{'Config':<20} {'Solved':<8} {'Evals':<8} {'Score':<8} {'OracleQ':<8}")
        for r in results:
            print(f"  {r['config']:<18} {str(r['solved']):<8} "
                  f"{r['evals']:<8} {r['final_score']:<8} {r['oracle_queries']:<8}")
        print(json.dumps(results, indent=2))
    else:
        result = run(args.variant, args.config, args.seed, args.max_evals, smart_prob)
        print(json.dumps(result, indent=2))
