#!/usr/bin/env python
"""Pre-registered analysis of results_ablation.jsonl.

Runs EXACTLY the 4 primary tests fixed in docs/QOVT_ABLATION_PREREGISTRATION.md
before any data existed. No test added or dropped after seeing results.
"""
import json
from collections import defaultdict
from scipy.stats import wilcoxon
import numpy as np

rows = [json.loads(l) for l in open("results_ablation.jsonl")]
by = defaultdict(dict)
for r in rows:
    by[(r["mode"], r["data"])][r["seed"]] = r["test_auc"]

SEEDS = list(range(42, 52))

def paired(mode_a, mode_b, data):
    a = np.array([by[(mode_a, data)][s] for s in SEEDS])
    b = np.array([by[(mode_b, data)][s] for s in SEEDS])
    diff = a - b
    stat, p = wilcoxon(a, b, alternative="greater")
    return dict(mean_a=a.mean(), mean_b=b.mean(), mean_delta=diff.mean(),
                n_pos=int((diff > 0).sum()), n=len(diff), raw_p=p, diffs=diff.tolist())

PRIMARY = [
    ("1", "quantum", "matched", "model_II"),
    ("2", "quantum", "matched", "model_III"),
    ("3", "quantum", "sham", "model_II"),
    ("4", "quantum", "sham", "model_III"),
]
SECONDARY = [
    ("5", "quantum", "classical", "model_II"),
    ("6", "quantum", "classical", "model_III"),
]

results = {}
print("=== PRIMARY (Holm-corrected over these 4) ===")
raw_ps = []
for tid, a, b, data in PRIMARY:
    r = paired(a, b, data)
    results[tid] = r
    raw_ps.append(r["raw_p"])
    print(f"[{tid}] {a}>{b} {data}: mean_a={r['mean_a']:.4f} mean_b={r['mean_b']:.4f} "
          f"delta={r['mean_delta']:+.4f} pos={r['n_pos']}/{r['n']} raw_p={r['raw_p']:.4f}")

# Holm-Bonferroni step-down correction
order = np.argsort(raw_ps)
m = len(raw_ps)
holm_p = [None] * m
running_max = 0.0
for rank, idx in enumerate(order):
    adj = raw_ps[idx] * (m - rank)
    running_max = max(running_max, adj)
    holm_p[idx] = min(running_max, 1.0)

print("\n=== Holm-adjusted p-values ===")
for (tid, a, b, data), hp in zip(PRIMARY, holm_p):
    r = results[tid]
    sig = "SIGNIFICANT (p<0.01)" if hp < 0.01 else "n.s."
    practical = (abs(r["mean_delta"]) >= 0.02 and r["n_pos"] >= 8 and r["mean_delta"] > 0)
    verdict = "CANDIDATE FINDING" if (hp < 0.01 and practical) else \
              ("directional only" if r["mean_delta"] > 0 else "no effect / reversed")
    print(f"[{tid}] {a}>{b} {data}: Holm p={hp:.4f} [{sig}] | delta={r['mean_delta']:+.4f} "
          f"pos={r['n_pos']}/10 | verdict={verdict}")

print("\n=== SECONDARY (reported, not part of primary claim, no correction applied) ===")
for tid, a, b, data in SECONDARY:
    r = paired(a, b, data)
    print(f"[{tid}] {a}>{b} {data}: mean_a={r['mean_a']:.4f} mean_b={r['mean_b']:.4f} "
          f"delta={r['mean_delta']:+.4f} pos={r['n_pos']}/{r['n']} raw_p={r['raw_p']:.4f}")

print("\n=== Arm means (all 4 arms, both datasets) ===")
for data in ["model_II", "model_III"]:
    for mode in ["quantum", "matched", "sham", "classical"]:
        vals = np.array([by[(mode, data)][s] for s in SEEDS])
        print(f"{data:10s} {mode:10s} mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")
