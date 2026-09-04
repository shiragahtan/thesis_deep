"""
plot.py
-------
Visualise results from one or more runs saved in results/.

Usage:
    python plot.py                          # plot all runs in results/
    python plot.py results/run_A.json results/run_B.json

Produces:
  1. Sample efficiency curve  (score vs evaluator calls)
  2. Smart vs dumb step breakdown per run
  3. Diversity over generations
"""

import json
import sys
import os
import glob
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── match presentation font if available ──────────────────────────────────────
for f in glob.glob(os.path.expanduser("~/Library/Fonts/Aptos*.ttf")):
    fm.fontManager.addfont(f)
matplotlib.rcParams['font.family'] = 'Aptos'

BG    = "#1A3A2A"; GOLD  = "#D4A843"; CREAM = "#F2EBD9"
GRN2  = "#162E1E"; MUTED = "#789678"; TEAL  = "#4ACFB0"
AMBER = "#E8953A"; RED   = "#C05050"; WARM_W= "#FFF8EE"

RUN_COLORS = [GOLD, TEAL, AMBER, RED, "#9B7FD4", "#60B8E0"]


def load_runs(paths: list) -> list:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append((os.path.basename(p), json.load(f)))
    return runs


def plot_all(paths: list, save_path: str = "results/summary_plot.png"):
    runs = load_runs(paths)
    if not runs:
        print("No run files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35)

    for ax in axes:
        ax.set_facecolor(GRN2)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.8)
        ax.tick_params(colors=CREAM, labelsize=9)

    # ── 1. Sample efficiency: score vs evaluator calls ────────────────────────
    ax = axes[0]
    ax.set_title("Sample Efficiency", color=WARM_W, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Evaluator calls", color=CREAM, fontsize=10)
    ax.set_ylabel("Best score",      color=CREAM, fontsize=10)
    ax.grid(True, color=MUTED, alpha=0.15)

    for (name, run), col in zip(runs, RUN_COLORS):
        gens  = run["generations"]
        # evaluator calls accumulate: 1 seed + NUM_PROPOSERS per gen (approx)
        # we reconstruct from proposers_tried
        evals = [1]
        for g in gens:
            evals.append(evals[-1] + g["proposers_tried"])
        scores = [gens[0]["best_score"]] + [g["best_score"] for g in gens]
        label  = name.replace("run_", "").replace(".json", "")
        ax.plot(evals, scores, color=col, lw=2.2, label=label)
        ax.scatter(evals[-1], scores[-1], color=col, s=50, zorder=5)

    if run.get("config", {}).get("TARGET_SCORE"):
        ax.axhline(run["config"]["TARGET_SCORE"], color=RED,
                   lw=1.4, linestyle='-.', alpha=0.7)
        ax.text(ax.get_xlim()[1] * 0.02, run["config"]["TARGET_SCORE"] + 1,
                "target", color=RED, fontsize=8)

    ax.legend(facecolor=GRN2, edgecolor=MUTED, labelcolor=CREAM, fontsize=9)

    # ── 2. Smart vs dumb step distribution ───────────────────────────────────
    ax = axes[1]
    ax.set_title("Smart vs Dumb Steps", color=WARM_W, fontsize=13, fontweight='bold', pad=8)
    ax.set_ylabel("Count", color=CREAM, fontsize=10)
    ax.grid(True, color=MUTED, alpha=0.15, axis='y')

    labels, smart_counts, dumb_counts = [], [], []
    for (name, run), col in zip(runs, RUN_COLORS):
        gens = run["generations"]
        smart = sum(1 for g in gens if g["step_type"] == "smart")
        dumb  = len(gens) - smart
        label = name.replace("run_", "").replace(".json", "")
        labels.append(label)
        smart_counts.append(smart)
        dumb_counts.append(dumb)

    x = range(len(labels))
    w = 0.35
    bars1 = ax.bar([i - w/2 for i in x], smart_counts, w, color=GOLD,  alpha=0.85, label="smart")
    bars2 = ax.bar([i + w/2 for i in x], dumb_counts,  w, color=MUTED, alpha=0.85, label="dumb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha='right', color=CREAM, fontsize=8)
    ax.legend(facecolor=GRN2, edgecolor=MUTED, labelcolor=CREAM, fontsize=9)

    # ── 3. Diversity over generations ─────────────────────────────────────────
    ax = axes[2]
    ax.set_title("Population Diversity", color=WARM_W, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Generation",     color=CREAM, fontsize=10)
    ax.set_ylabel("Score std-dev",  color=CREAM, fontsize=10)
    ax.grid(True, color=MUTED, alpha=0.15)

    for (name, run), col in zip(runs, RUN_COLORS):
        gens  = run["generations"]
        gen_x = [g["generation"] for g in gens]
        divs  = [g["diversity"]  for g in gens]
        label = name.replace("run_", "").replace(".json", "")
        ax.plot(gen_x, divs, color=col, lw=2.0, label=label)

    ax.legend(facecolor=GRN2, edgecolor=MUTED, labelcolor=CREAM, fontsize=9)

    fig.suptitle("LLM-Guided Evolutionary Search — Results",
                 color=WARM_W, fontsize=15, fontweight='bold', y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("results/run_*.json"))

    if not paths:
        print("No result files found. Run main.py first.")
    else:
        print(f"Plotting {len(paths)} run(s): {paths}")
        plot_all(paths)
