#!/usr/bin/env python
"""Generate result figures for the pre-registered ablation (Round 1 RY+CNOT,
Round 2 Butterfly) for docs/QOVT_PAPER.tex. Reads only from the two verified
results files -- no numbers invented."""
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = list(range(42, 52))
by = defaultdict(dict)
for r in [json.loads(l) for l in open("results_ablation.jsonl")]:
    key = (r["mode"] if r["mode"] != "quantum" else "rycnot", r["data"])
    by[key][r["seed"]] = r["test_auc"]
for r in [json.loads(l) for l in open("results_butterfly.jsonl")]:
    by[("butterfly", r["data"])][r["seed"]] = r["test_auc"]

def stats(mode, data):
    v = np.array([by[(mode, data)][s] for s in SEEDS])
    return v.mean(), v.std()

ARMS = ["rycnot", "butterfly", "matched", "sham", "classical"]
LABELS = {"rycnot": "RY+CNOT\n(quantum)", "butterfly": "Givens butterfly\n(classical)",
          "matched": "matched\n(rank-1)", "sham": "sham\n(unconstrained)",
          "classical": "classical\n(MHA)"}
COLORS = {"rycnot": "#d97706", "butterfly": "#d97706", "matched": "#6b7280",
          "sham": "#6b7280", "classical": "#7c3aed"}
HATCH  = {"rycnot": "", "butterfly": "//", "matched": "", "sham": "\\\\", "classical": ""}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
for ax, data, title in zip(axes, ["model_II", "model_III"], ["Model_II", "Model_III"]):
    means = [stats(a, data)[0] for a in ARMS]
    stds  = [stats(a, data)[1] for a in ARMS]
    x = np.arange(len(ARMS))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                   color=[COLORS[a] for a in ARMS], hatch=[HATCH[a] for a in ARMS],
                   edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=8)
    ax.set_title(f"{title}, $N{{=}}500$/class, $n{{=}}10$ seeds", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.025, f"{m:.3f}", ha="center", fontsize=7.5)
axes[0].set_ylabel("Test AUC (held-out, macro OVR)")
fig.suptitle("Pre-registered ablation, Round 1 (RY+CNOT) + Round 2 (Givens butterfly)"
             " -- mean $\\pm$ std over 10 seeds", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("../../docs/figures/qovt_ablation_results.pdf", dpi=200)
fig.savefig("../../docs/figures/qovt_ablation_results.png", dpi=200)
print("wrote docs/figures/qovt_ablation_results.{pdf,png}")

# per-seed dot plot for the two RY+CNOT vs sham cells (shows the 7/10 vs 10/10 pattern)
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, data, title in zip(axes2, ["model_II", "model_III"], ["Model_II (significant)", "Model_III (n.s.)"]):
    q = np.array([by[("rycnot", data)][s] for s in SEEDS])
    s = np.array([by[("sham", data)][s] for s in SEEDS])
    for i, seed in enumerate(SEEDS):
        color = "#16a34a" if q[i] > s[i] else "#dc2626"
        ax.plot([0, 1], [q[i], s[i]], color=color, alpha=0.6, linewidth=1.2, marker="o", markersize=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RY+CNOT", "sham"])
    ax.set_title(title, fontsize=10)
axes2[0].set_ylabel("Test AUC (per-seed pair)")
fig2.suptitle("RY+CNOT vs sham, paired by seed (green = quantum wins that seed)", fontsize=10)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
fig2.savefig("../../docs/figures/qovt_paired_seeds.pdf", dpi=200)
fig2.savefig("../../docs/figures/qovt_paired_seeds.png", dpi=200)
print("wrote docs/figures/qovt_paired_seeds.{pdf,png}")
