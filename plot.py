"""
plot.py
-------
Generate all thesis plots from saved result JSON files.

Usage:
    python3 plot.py

Outputs (saved to figures/):
  - fig1_score_vs_evals_primes.png   — smart vs dumb on primes (the headline result)
  - fig2_dumb_multiseed.png          — dumb-only consistency across 4 seeds x 3 benchmarks
  - fig3_benchmark_comparison.png    — final scores bar chart across all benchmarks
"""

import json
import os
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Known result files (keyed by (benchmark, mode, seed)) ─────────────────────
# Map from (benchmark, mode, seed) → result file path
# Add new files here as experiments complete.

RESULT_MAP = {
    # Primes
    ("primes",   "smart", 42): "results/run_1788848147.json",
    ("primes",   "dumb",  42): "results/run_1788848097.json",
    # Sorting
    ("sorting",  "smart", 42): "results/run_1788848030.json",
    ("sorting",  "dumb",  42): "results/run_1788845672.json",
    # Strings
    ("strings",  "dumb",  42): "results/run_1788848480.json",
    ("strings",  "smart", 42): "results/run_1788980054.json",
    # Matrix
    ("matrix",   "dumb",  42): "results/run_1788962853.json",
    # Mixed config
    ("primes",   "mixed", 42): "results/run_1788980442.json",
}

# Dumb multi-seed results (scores only — from terminal output, no JSON per-seed yet)
DUMB_MULTISEED = {
    "sorting": {"seeds": [42, 1, 7, 99], "scores": [74.2, 72.2, 73.0, 71.8], "evals": [100, 99, 97, 98]},
    "primes":  {"seeds": [42, 1, 7, 99], "scores": [75.3, 75.0, 75.5, 75.3], "evals": [77, 77, 76, 74]},
    "strings": {"seeds": [42, 1, 7, 99], "scores": [74.3, 73.7, 73.4, 72.8], "evals": [81, 84, 88, 79]},
}

COLORS = {
    "smart": "#2E86AB",   # blue
    "dumb":  "#E84855",   # red
    "mixed": "#F4A261",   # orange
}

os.makedirs("figures", exist_ok=True)

# ── Helper ─────────────────────────────────────────────────────────────────────

def load(key):
    path = RESULT_MAP.get(key)
    if path and os.path.exists(path):
        return json.load(open(path))
    return None

def score_curve(result):
    """Return (cumulative_evals, best_score) arrays from a result dict."""
    gens = result.get("generations", [])
    evals, scores = [0], [result["generations"][0]["best_score"] if gens else 70]
    cumulative = 0
    for g in gens:
        cumulative += g.get("proposers_tried", 2)
        evals.append(cumulative)
        scores.append(g["best_score"])
    return np.array(evals), np.array(scores)

# ── Figure 1: Smart vs Dumb on Primes (headline result) ───────────────────────

def fig1_primes():
    smart = load(("primes", "smart", 42))
    dumb  = load(("primes", "dumb",  42))
    if not smart or not dumb:
        print("fig1: missing data, skipping"); return

    fig, ax = plt.subplots(figsize=(8, 5))

    se, ss = score_curve(smart)
    de, ds = score_curve(dumb)

    ax.plot(se, ss, color=COLORS["smart"], linewidth=2.5, marker='o', markersize=4,
            label=f"Smart-only (LLM-guided)  →  {smart['best_score']:.0f}/100 in {smart['total_evaluations']} evals")
    ax.plot(de, ds, color=COLORS["dumb"],  linewidth=2.5, marker='s', markersize=4,
            label=f"Dumb-only (random)        →  {dumb['best_score']:.1f}/100 in {dumb['total_evaluations']} evals")

    ax.axhline(95, color="gray", linestyle="--", linewidth=1.2, label="Target score (95)")
    ax.axhline(75.3, color=COLORS["dumb"], linestyle=":", linewidth=1, alpha=0.6)

    # Annotate the 8.5× callout
    ax.annotate("100/100\n(9 evals)", xy=(smart['total_evaluations'], smart['best_score']),
                xytext=(20, -15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS["smart"]),
                color=COLORS["smart"], fontsize=10, fontweight='bold')
    ax.annotate("Stuck at 75.3\n(77 evals)", xy=(dumb['total_evaluations'], dumb['best_score']),
                xytext=(10, 10), textcoords='offset points',
                color=COLORS["dumb"], fontsize=9)

    ax.set_xlabel("Evaluations (program executions)", fontsize=12)
    ax.set_ylabel("Best score / 100", fontsize=12)
    ax.set_title("Prime Number Generation: Smart vs Dumb Step\n"
                 r"$\bf{8.5\times}$ improvement in sample efficiency", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig1_score_vs_evals_primes.png", dpi=150)
    plt.close()
    print("✓ figures/fig1_score_vs_evals_primes.png")

# ── Figure 2: Dumb-only multi-seed consistency ─────────────────────────────────

def fig2_dumb_multiseed():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    benchmarks = ["sorting", "primes", "strings"]
    titles = ["Sorting", "Prime Generation", "String Search"]

    for ax, bm, title in zip(axes, benchmarks, titles):
        data = DUMB_MULTISEED[bm]
        seeds = data["seeds"]
        scores = data["scores"]
        evals  = data["evals"]
        mean_s = np.mean(scores)
        std_s  = np.std(scores)

        bars = ax.bar([str(s) for s in seeds], scores,
                      color=COLORS["dumb"], alpha=0.75, edgecolor='black', linewidth=0.8)
        ax.axhline(95, color="gray", linestyle="--", linewidth=1.2, label="Target (95)")
        ax.axhline(mean_s, color="darkred", linestyle="-", linewidth=1.5,
                   label=f"Mean={mean_s:.1f}")

        ax.set_title(f"{title}\nmean={mean_s:.1f} ± {std_s:.1f}", fontsize=11)
        ax.set_xlabel("Random seed", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Best score / 100", fontsize=11)
        ax.set_ylim(60, 102)
        ax.legend(fontsize=8)

        # Label bars with eval counts
        for bar, ev in zip(bars, evals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{ev}ev", ha='center', va='bottom', fontsize=8, color='#333')

    fig.suptitle("Dumb-Only Baseline: Consistent Ceiling Across 4 Seeds\n"
                 "(target=95 never reached — strategy limited, not seed-dependent)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig2_dumb_multiseed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ figures/fig2_dumb_multiseed.png")

# ── Figure 3: Final score comparison bar chart ─────────────────────────────────

def fig3_comparison():
    # Known results: (benchmark, smart_score, dumb_avg_score)
    data = [
        ("Sorting",  71.9, 72.8),
        ("Primes",  100.0, 75.3),
        ("Strings",  97.4, 73.6),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(data))
    w = 0.35

    dumb_scores  = [d[2] for d in data]
    smart_scores = [d[1] if d[1] is not None else 0 for d in data]
    labels       = [d[0] for d in data]

    b1 = ax.bar(x - w/2, dumb_scores,  w, label="Dumb-only (avg 4 seeds)", color=COLORS["dumb"],  alpha=0.8)
    b2 = ax.bar(x + w/2, smart_scores, w, label="Smart-only (seed 42)",     color=COLORS["smart"], alpha=0.8)

    ax.axhline(95, color="gray", linestyle="--", linewidth=1.5, label="Target (95)")

    # Annotate the key result
    ax.annotate("8.5× fewer\nevaluations!", xy=(1 + w/2, 100),
                xytext=(1 + w/2 + 0.3, 93),
                arrowprops=dict(arrowstyle='->', color=COLORS["smart"]),
                color=COLORS["smart"], fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Best score / 100", fontsize=12)
    ax.set_title("Smart vs Dumb Step — Final Scores by Benchmark", fontsize=13)
    ax.set_ylim(60, 108)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig3_benchmark_comparison.png", dpi=150)
    plt.close()
    print("✓ figures/fig3_benchmark_comparison.png")

# ── Figure 4: Score trajectory on sorting (shows diversity effect) ─────────────

def fig4_sorting():
    smart = load(("sorting", "smart", 42))
    dumb  = load(("sorting", "dumb",  42))
    if not smart or not dumb:
        print("fig4: missing data, skipping"); return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    # Score curves
    se, ss = score_curve(smart)
    de, ds = score_curve(dumb)
    ax1.plot(se, ss, color=COLORS["smart"], linewidth=2, label=f"Smart → {smart['best_score']:.1f}/100")
    ax1.plot(de, ds, color=COLORS["dumb"],  linewidth=2, label=f"Dumb  → {dumb['best_score']:.1f}/100")
    ax1.axhline(95, color="gray", linestyle="--", linewidth=1, label="Target (95)")
    ax1.set_ylabel("Best score / 100")
    ax1.set_title("Sorting Benchmark: Smart vs Dumb\n(ceiling effect — C Timsort baseline)")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(60, 100)

    # Diversity comparison
    s_div = [g["diversity"] for g in smart.get("generations", [])]
    d_div = [g["diversity"] for g in dumb.get("generations", [])]
    ax2.plot(range(1, len(s_div)+1), s_div, color=COLORS["smart"], linewidth=2,
             label="Smart diversity (std-dev of scores)")
    ax2.plot(range(1, len(d_div)+1), d_div, color=COLORS["dumb"],  linewidth=2,
             label="Dumb diversity")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Population diversity\n(score std-dev)")
    ax2.set_title("Diversity: Smart step avoids collapse, Dumb collapses to ~0")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig4_sorting_diversity.png", dpi=150)
    plt.close()
    print("✓ figures/fig4_sorting_diversity.png")

# ── Figure 5: Smart vs Dumb on Strings ────────────────────────────────────────

def fig5_strings():
    smart = load(("strings", "smart", 42))
    dumb  = load(("strings", "dumb",  42))
    if not smart or not dumb:
        print("fig5: missing data, skipping"); return

    fig, ax = plt.subplots(figsize=(8, 5))

    se, ss = score_curve(smart)
    de, ds = score_curve(dumb)

    ax.plot(se, ss, color=COLORS["smart"], linewidth=2.5, marker='o', markersize=4,
            label=f"Smart-only  →  {smart['best_score']:.1f}/100 in {smart['total_evaluations']} evals")
    ax.plot(de, ds, color=COLORS["dumb"],  linewidth=2.5, marker='s', markersize=4,
            label=f"Dumb-only   →  {dumb['best_score']:.1f}/100 in {dumb['total_evaluations']} evals")

    ax.axhline(95, color="gray", linestyle="--", linewidth=1.2, label="Target (95)")

    ax.annotate("97.4/100\n(51 evals)", xy=(smart['total_evaluations'], smart['best_score']),
                xytext=(-40, -18), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS["smart"]),
                color=COLORS["smart"], fontsize=10, fontweight='bold')
    ax.annotate("Stuck at 74.3\n(81 evals)", xy=(dumb['total_evaluations'], dumb['best_score']),
                xytext=(5, 8), textcoords='offset points',
                color=COLORS["dumb"], fontsize=9)

    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Best score / 100", fontsize=12)
    ax.set_title("String Search: Smart vs Dumb Step\n"
                 "LLM discovers str.find() loop — 1.6× more sample-efficient", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig5_score_vs_evals_strings.png", dpi=150)
    plt.close()
    print("✓ figures/fig5_score_vs_evals_strings.png")

# ── Figure 6: All three configs on primes (the key 3-way comparison) ──────────

def fig6_primes_threeway():
    smart = load(("primes", "smart", 42))
    dumb  = load(("primes", "dumb",  42))
    mixed = load(("primes", "mixed", 42))
    if not smart or not dumb or not mixed:
        print("fig6: missing data, skipping"); return

    fig, ax = plt.subplots(figsize=(9, 5))

    se, ss = score_curve(smart)
    de, ds = score_curve(dumb)
    me, ms = score_curve(mixed)

    ax.plot(se, ss, color=COLORS["smart"], linewidth=2.5, marker='o', markersize=5,
            label=f"Smart-only (100% LLM)    →  {smart['best_score']:.0f}/100 in {smart['total_evaluations']} evals")
    ax.plot(me, ms, color=COLORS["mixed"], linewidth=2.5, marker='^', markersize=5,
            label=f"Mixed (70% smart+30% dumb)  →  {mixed['best_score']:.0f}/100 in {mixed['total_evaluations']} evals")
    ax.plot(de, ds, color=COLORS["dumb"],  linewidth=2.5, marker='s', markersize=5,
            label=f"Dumb-only (0% LLM)       →  {dumb['best_score']:.1f}/100 in {dumb['total_evaluations']} evals (no target)")

    ax.axhline(95, color="gray", linestyle="--", linewidth=1.2, label="Target (95)")

    ax.annotate(f"{smart['best_score']:.0f}/100\n({smart['total_evaluations']} evals)",
                xy=(smart['total_evaluations'], smart['best_score']),
                xytext=(8, -18), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS["smart"]),
                color=COLORS["smart"], fontsize=9, fontweight='bold')
    ax.annotate(f"{mixed['best_score']:.0f}/100\n({mixed['total_evaluations']} evals)",
                xy=(mixed['total_evaluations'], mixed['best_score']),
                xytext=(8, 5), textcoords='offset points',
                color=COLORS["mixed"], fontsize=9, fontweight='bold')

    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Best score / 100", fontsize=12)
    ax.set_title("Prime Number Generation: All Three Configs\n"
                 "Smart-only fastest; mixed also solves it; dumb never does", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(60, 108)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig6_primes_threeway.png", dpi=150)
    plt.close()
    print("✓ figures/fig6_primes_threeway.png")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating thesis figures...")
    fig1_primes()
    fig2_dumb_multiseed()
    fig3_comparison()
    fig4_sorting()
    fig5_strings()
    fig6_primes_threeway()
    print("\nAll figures saved to figures/")
