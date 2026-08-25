#!/usr/bin/env python3
"""Pre-registered analysis for QPE-2 (docs/QPE2_PREREGISTRATION.md,
committed 402781b before any sweep data existed).

Primary family (Holm over 6, one-sided paired Wilcoxon, n=20, N=100):
  1,2 quantum > dctfix  (does learnable refinement beat the fixed prior)
  3,4 quantum > skew48  (circuit chart vs PARAM-MATCHED classical chart)
  5,6 quantum > cayley  (circuit chart vs full-dim classical chart)
Headline claim requires tests 1-4 to pass the practical bar
(Holm p<0.01 AND |delta|>=2pp AND >=16/20 seeds).

Secondary (uncorrected): same six at N=250; quantum vs butterfly;
quantum vs conv; TOST-style equivalence (90% CI within +/-2pp).
"""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, t as tdist

SEEDS = list(range(42, 62))
DATASETS = ["model_II", "model_III"]
ARMS = ["quantum", "skew48", "butterfly", "cayley", "dctfix", "conv"]
NS = [100, 250]

by, drift = defaultdict(dict), defaultdict(dict)
for ds in DATASETS:
    for line in open(f"results_qpe2_{ds}.jsonl"):
        r = json.loads(line)
        key = (r["embed"], r["data"], r["n_per_class"])
        if r["seed"] in by[key]:
            raise SystemExit(f"DUPLICATE seed {r['seed']} for {key} -- "
                             "stale results file, rerun the sweep")
        by[key][r["seed"]] = r["test_auc"]
        if r.get("R_frob_drift") is not None:
            drift[key][r["seed"]] = r["R_frob_drift"]

def vec(arm, ds, n):
    return np.array([by[(arm, ds, n)][s] for s in SEEDS])

def paired(a, b, ds, n, alt="greater"):
    av, bv = vec(a, ds, n), vec(b, ds, n)
    d = av - bv
    _, p = wilcoxon(av, bv, alternative=alt)
    return dict(mean_a=av.mean(), mean_b=bv.mean(), delta=d.mean(),
                n_pos=int((d > 0).sum()), raw_p=p, diff=d)

def holm(ps):
    order = np.argsort(ps); m = len(ps)
    out = [None] * m; run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, ps[idx] * (m - rank))
        out[idx] = min(run, 1.0)
    return out

print("=== Arm means (n=20 seeds, held-out test AUC) ===")
for n in NS:
    for ds in DATASETS:
        for arm in ARMS:
            v = vec(arm, ds, n)
            print(f"N={n:3d} {ds:10s} {arm:10s} mean={v.mean():.4f} "
                  f"std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")

PRIMARY = [("1", "quantum", "dctfix", "model_II"),
           ("2", "quantum", "dctfix", "model_III"),
           ("3", "quantum", "skew48", "model_II"),
           ("4", "quantum", "skew48", "model_III"),
           ("5", "quantum", "cayley", "model_II"),
           ("6", "quantum", "cayley", "model_III")]

print("\n=== PRIMARY (N=100, one-sided greater, Holm over 6) ===")
res = {tid: paired(a, b, ds, 100) for tid, a, b, ds in PRIMARY}
hps = holm([res[tid]["raw_p"] for tid, *_ in PRIMARY])
passed = {}
for (tid, a, b, ds), hp in zip(PRIMARY, hps):
    r = res[tid]
    sig = hp < 0.01
    ok = sig and r["delta"] >= 0.02 and r["n_pos"] >= 16
    passed[tid] = ok
    verdict = ("PASSES practical bar" if ok else
               "significant, fails 2pp/seed bar" if sig else
               "directional only" if r["delta"] > 0 else "no effect / reversed")
    print(f"[{tid}] {a}>{b} {ds}: mean {r['mean_a']:.4f} vs {r['mean_b']:.4f} "
          f"delta={r['delta']:+.4f} pos={r['n_pos']}/20 raw_p={r['raw_p']:.5f} "
          f"Holm={hp:.5f} | {verdict}")

headline = all(passed[t] for t in ["1", "2", "3", "4"])
print(f"\n>>> HEADLINE (tests 1-4 all pass): "
      f"{'YES -- pre-registered small-data quantum win' if headline else 'NO'}")
if not headline and all(passed[t] for t in ["5", "6"]) and not (passed["3"] and passed["4"]):
    print(">>> Beat cayley but not skew48: the effect is PARAMETER COUNT, "
          "not chart -- ordinary statistics, no quantum content "
          "(pre-committed reading).")

print("\n=== SECONDARY: same six at N=250 (uncorrected) ===")
for tid, a, b, ds in PRIMARY:
    r = paired(a, b, ds, 250)
    print(f"[{tid}'] {a}>{b} {ds}: delta={r['delta']:+.4f} "
          f"pos={r['n_pos']}/20 raw_p={r['raw_p']:.5f}")

print("\n=== SECONDARY: quantum vs butterfly / conv (uncorrected) ===")
for other in ["butterfly", "conv"]:
    for n in NS:
        for ds in DATASETS:
            r = paired("quantum", other, ds, n)
            print(f"N={n:3d} {ds:10s} quantum>{other}: delta={r['delta']:+.4f} "
                  f"pos={r['n_pos']}/20 raw_p={r['raw_p']:.5f}")

print("\n=== Equivalence (TOST-style: 90% CI within +/-2pp) ===")
for n in NS:
    for ds in DATASETS:
        for other in ["dctfix", "skew48"]:
            d = paired("quantum", other, ds, n, "two-sided")["diff"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            tc = tdist.ppf(0.95, len(d) - 1)
            lo, hi = d.mean() - tc * se, d.mean() + tc * se
            eq = lo > -0.02 and hi < 0.02
            print(f"N={n:3d} {ds:10s} quantum vs {other:7s}: "
                  f"delta={d.mean():+.4f} 90%CI=[{lo:+.4f},{hi:+.4f}] -> "
                  f"{'EQUIVALENT' if eq else 'not equivalent/inconclusive'}")

print("\n=== Chart-drift diagnostic (did each chart actually move?) ===")
for n in NS:
    for ds in DATASETS:
        for arm in ["quantum", "skew48", "butterfly", "cayley"]:
            v = np.array([drift[(arm, ds, n)][s] for s in SEEDS])
            print(f"N={n:3d} {ds:10s} {arm:10s} R_frob_drift mean={v.mean():.3f} "
                  f"min={v.min():.3f} max={v.max():.3f}  (||R||_F=8)")
