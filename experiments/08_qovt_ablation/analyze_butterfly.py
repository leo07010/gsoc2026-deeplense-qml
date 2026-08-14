#!/usr/bin/env python
"""Pre-registered analysis of the butterfly-circuit round (Round 2 in
docs/QOVT_ABLATION_PREREGISTRATION.md). Butterfly quantum arm from
results_butterfly.jsonl; matched/sham/classical/RY+CNOT-quantum reused from
results_ablation.jsonl (job 251067) -- see the pre-registration doc for why
that reuse is valid (those controls don't depend on quantum circuit choice).
"""
import json
from collections import defaultdict
from scipy.stats import wilcoxon
import numpy as np

SEEDS = list(range(42, 52))

by = defaultdict(dict)
for r in [json.loads(l) for l in open("results_ablation.jsonl")]:
    key = (r["mode"] if r["mode"] != "quantum" else "rycnot", r["data"])
    by[key][r["seed"]] = r["test_auc"]
for r in [json.loads(l) for l in open("results_butterfly.jsonl")]:
    by[("butterfly", r["data"])][r["seed"]] = r["test_auc"]

def paired(a, b, data, alt="greater"):
    av = np.array([by[(a, data)][s] for s in SEEDS])
    bv = np.array([by[(b, data)][s] for s in SEEDS])
    diff = av - bv
    stat, p = wilcoxon(av, bv, alternative=alt)
    return dict(mean_a=av.mean(), mean_b=bv.mean(), mean_delta=diff.mean(),
                n_pos=int((diff > 0).sum()), n=len(diff), raw_p=p)

PRIMARY = [
    ("1", "butterfly", "matched", "model_II"),
    ("2", "butterfly", "matched", "model_III"),
    ("3", "butterfly", "sham", "model_II"),
    ("4", "butterfly", "sham", "model_III"),
]
SECONDARY_ONESIDED = [
    ("5", "butterfly", "classical", "model_II"),
    ("6", "butterfly", "classical", "model_III"),
]
SECONDARY_TWOSIDED = [
    ("7", "butterfly", "rycnot", "model_II"),
    ("8", "butterfly", "rycnot", "model_III"),
]

print("=== PRIMARY (Holm-corrected over these 4) ===")
results, raw_ps = {}, []
for tid, a, b, data in PRIMARY:
    r = paired(a, b, data)
    results[tid] = r
    raw_ps.append(r["raw_p"])
    print(f"[{tid}] {a}>{b} {data}: mean_a={r['mean_a']:.4f} mean_b={r['mean_b']:.4f} "
          f"delta={r['mean_delta']:+.4f} pos={r['n_pos']}/{r['n']} raw_p={r['raw_p']:.4f}")

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
    sig = hp < 0.01
    practical = (abs(r["mean_delta"]) >= 0.02 and r["n_pos"] >= 8 and r["mean_delta"] > 0)
    verdict = "CANDIDATE FINDING" if (sig and practical) else \
              ("directional only" if r["mean_delta"] > 0 else "no effect / reversed")
    print(f"[{tid}] {a}>{b} {data}: Holm p={hp:.4f} [{'SIG' if sig else 'n.s.'}] "
          f"delta={r['mean_delta']:+.4f} pos={r['n_pos']}/10 | verdict={verdict}")

print("\n=== SECONDARY: butterfly > classical (one-sided, no correction) ===")
for tid, a, b, data in SECONDARY_ONESIDED:
    r = paired(a, b, data)
    print(f"[{tid}] {a}>{b} {data}: delta={r['mean_delta']:+.4f} pos={r['n_pos']}/{r['n']} raw_p={r['raw_p']:.4f}")

print("\n=== SECONDARY: butterfly vs RY+CNOT (TWO-SIDED, paired by seed) ===")
for tid, a, b, data in SECONDARY_TWOSIDED:
    r = paired(a, b, data, alt="two-sided")
    print(f"[{tid}] {a} vs {b} {data}: mean_butterfly={r['mean_a']:.4f} mean_rycnot={r['mean_b']:.4f} "
          f"delta={r['mean_delta']:+.4f} pos={r['n_pos']}/{r['n']} raw_p(2-sided)={r['raw_p']:.4f}")

print("\n=== Arm means ===")
for data in ["model_II", "model_III"]:
    for mode in ["butterfly", "rycnot", "matched", "sham", "classical"]:
        vals = np.array([by[(mode, data)][s] for s in SEEDS])
        print(f"{data:10s} {mode:10s} mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")
