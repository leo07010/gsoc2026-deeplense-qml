#!/usr/bin/env python3
"""Pre-registered analysis for the QPE experiment (docs/QPE_PREREGISTRATION.md,
commit 806ec8c, committed before any sweep data existed).

Primary family (Holm over 4, one-sided paired Wilcoxon, n=20 seeds):
  1 quantum > scramble  model_II    (circuit bit-structure)
  2 quantum > scramble  model_III
  3 quantum > conv      model_II    (inductive bias vs learned embed)
  4 quantum > conv      model_III
Practical bar: Holm p<0.01 AND |mean delta|>=2pp AND >=16/20 seeds.

Secondary:
  quantum vs dct  two-sided; equivalence ONLY via TOST-style rule
    (90% CI of paired mean delta entirely within +/-2pp).
  scramble vs conv two-sided (context).

Also reports arm means/stds and the angle-drift diagnostic.
"""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, t as tdist

SEEDS = list(range(42, 62))
DATASETS = ["model_II", "model_III"]
ARMS = ["quantum", "scramble", "dct", "conv"]

by = defaultdict(dict)
drift = defaultdict(dict)
for ds in DATASETS:
    for r in [json.loads(l) for l in open(f"results_qpe_{ds}.jsonl")]:
        by[(r["embed"], r["data"])][r["seed"]] = r["test_auc"]
        if r.get("U_frob_drift") is not None:
            drift[(r["embed"], r["data"])][r["seed"]] = r["U_frob_drift"]

def vec(arm, ds):
    return np.array([by[(arm, ds)][s] for s in SEEDS])

print("=== Arm means (n=20 seeds, held-out test AUC) ===")
for ds in DATASETS:
    for arm in ARMS:
        v = vec(arm, ds)
        print(f"{ds:10s} {arm:9s} mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")

def paired(a, b, ds, alt):
    av, bv = vec(a, ds), vec(b, ds)
    diff = av - bv
    stat, p = wilcoxon(av, bv, alternative=alt)
    return dict(mean_a=av.mean(), mean_b=bv.mean(), delta=diff.mean(),
                n_pos=int((diff > 0).sum()), raw_p=p, diff=diff)

PRIMARY = [
    ("1", "quantum", "scramble", "model_II"),
    ("2", "quantum", "scramble", "model_III"),
    ("3", "quantum", "conv",     "model_II"),
    ("4", "quantum", "conv",     "model_III"),
]

print("\n=== PRIMARY (one-sided greater, Holm over 4) ===")
res, raw_ps = {}, []
for tid, a, b, ds in PRIMARY:
    r = paired(a, b, ds, "greater")
    res[tid] = r; raw_ps.append(r["raw_p"])
    print(f"[{tid}] {a}>{b} {ds}: mean_a={r['mean_a']:.4f} mean_b={r['mean_b']:.4f} "
          f"delta={r['delta']:+.4f} pos={r['n_pos']}/20 raw_p={r['raw_p']:.5f}")

order = np.argsort(raw_ps); m = len(raw_ps)
holm = [None]*m; running = 0.0
for rank, idx in enumerate(order):
    running = max(running, raw_ps[idx]*(m-rank)); holm[idx] = min(running, 1.0)

print("\n=== Holm-adjusted verdicts (bar: p<0.01 AND |d|>=2pp AND >=16/20) ===")
for (tid, a, b, ds), hp in zip(PRIMARY, holm):
    r = res[tid]
    sig = hp < 0.01
    practical = sig and abs(r["delta"]) >= 0.02 and r["n_pos"] >= 16 and r["delta"] > 0
    verdict = ("CONFIRMED" if practical else
               "significant but fails practical bar" if sig else
               "directional only" if r["delta"] > 0 else "no effect / reversed")
    print(f"[{tid}] {a}>{b} {ds}: Holm p={hp:.5f} [{'SIG' if sig else 'n.s.'}] "
          f"delta={r['delta']:+.4f} pos={r['n_pos']}/20 | {verdict}")

print("\n=== SECONDARY: quantum vs dct (two-sided + TOST-style CI rule) ===")
for ds in DATASETS:
    r = paired("quantum", "dct", ds, "two-sided")
    d = r["diff"]; n = len(d)
    se = d.std(ddof=1)/np.sqrt(n)
    tcrit = tdist.ppf(0.95, n-1)          # 90% CI
    lo, hi = d.mean()-tcrit*se, d.mean()+tcrit*se
    equiv = (lo > -0.02) and (hi < 0.02)
    print(f"{ds}: delta={d.mean():+.4f} raw_p(2s)={r['raw_p']:.5f} "
          f"90%CI=[{lo:+.4f},{hi:+.4f}] -> "
          f"{'EQUIVALENT (within +/-2pp)' if equiv else 'NOT equivalent / inconclusive'}")

print("\n=== SECONDARY: scramble vs conv (two-sided, context) ===")
for ds in DATASETS:
    r = paired("scramble", "conv", ds, "two-sided")
    print(f"{ds}: delta={r['delta']:+.4f} pos={r['n_pos']}/20 raw_p={r['raw_p']:.5f}")

print("\n=== Angle-drift diagnostic (did the circuit engage?) ===")
for ds in DATASETS:
    for arm in ["quantum", "scramble"]:
        v = np.array([drift[(arm, ds)][s] for s in SEEDS])
        print(f"{ds:10s} {arm:9s} U_frob_drift mean={v.mean():.3f} "
              f"min={v.min():.3f} max={v.max():.3f}  (identity-scale ref: ||U||_F=8)")
