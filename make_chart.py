"""Draw results/chart.png from results/summary.csv. Run after experiments.py: python make_chart.py"""
import csv

import matplotlib.pyplot as plt

METHODS = [  # (name in summary.csv, legend label, color) - fixed order, color follows the method
    ("bm25", "BM25", "#2a78d6"),
    ("semantic", "Semantic", "#eb6834"),
    ("semantic-win", "Semantic, windowed", "#1baf7a"),
    ("hybrid", "Hybrid", "#eda100"),
    ("hybrid-win", "Hybrid, windowed", "#e87ba4"),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

with open("results/summary.csv", encoding="utf-8") as f:
    rows = {(int(r["chunk_size"]), r["method"]): r for r in csv.DictReader(f)}
sizes = sorted({size for size, _ in rows})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bar_w = 0.15
for m, (method, label, color) in enumerate(METHODS):
    xs = [i + (m - 2) * bar_w for i in range(len(sizes))]
    mrr = [float(rows[(s, method)]["rr"]) for s in sizes]
    err_low = [v - float(rows[(s, method)]["rr_ci_low"]) for v, s in zip(mrr, sizes)]
    err_high = [float(rows[(s, method)]["rr_ci_high"]) - v for v, s in zip(mrr, sizes)]
    # bar width minus a small gap, so neighbouring bars don't touch
    ax.bar(xs, mrr, width=bar_w - 0.02, color=color, label=label, zorder=2)
    ax.errorbar(xs, mrr, yerr=[err_low, err_high], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=3)

# Random baseline: one dashed line across each group
for i, s in enumerate(sizes):
    rnd = float(rows[(s, "bm25")]["random_rr"])
    ax.hlines(rnd, i - 2.5 * bar_w, i + 2.5 * bar_w, colors=MUTED, linestyles="--", linewidth=1.2, zorder=4)
ax.plot([], [], color=MUTED, linestyle="--", linewidth=1.2, label="Random baseline")

ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"{s}-word chunks\n({rows[(s, 'bm25')]['n_chunks']} chunks)" for s in sizes], color=INK)
ax.set_ylabel("MRR (higher is better)", color=INK)
ax.set_ylim(0, 1)
ax.set_title("Retrieval quality by method and chunk size (43 questions, 95% bootstrap CI)",
             color=INK, fontsize=10, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.tick_params(colors=MUTED, length=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left", labelcolor=INK)

fig.tight_layout()
fig.savefig("results/chart.png", facecolor=SURFACE)
print("Saved results/chart.png")
