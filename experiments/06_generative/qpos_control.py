#!/usr/bin/env python
"""
POSITIVE CONTROL for the whole null-result study (the experiment that gives
the nulls their teeth). A bidirectional, symmetric demonstration:

  target = IQP Born distribution (quantum-native)  → does the IQP model win?
  target = Ising distribution     (classical-native) → does Ising win, IQP lose?

Same model families, same code, same optimizer as qgen_iqp.py — only the
DATA-GENERATING PROCESS changes. Exact enumeration (n bits → 2^n states),
exact KL minimization (infinite-data / expressivity ceiling).

Reported: excess NLL = KL(p_target || p_model) in nats (0 = exact recovery).
Swept over a phase/coupling scale s that controls how "structured" the target
is (s→0 ⇒ near-uniform ⇒ everyone ties; larger s ⇒ structure turns on).

Interpretation:
  • IQP wins on the IQP target, loses on the Ising target, and (from qgen_iqp)
    loses on real lensing latents ⇒ the quantum model is correctly built and
    trainable; the natural-data null is REAL, not an artifact.
  • Which model wins is a function of the data-generating process. Natural
    lensing latents are Ising-like (low order) ⇒ classical wins there.
"""
import os, argparse
import numpy as np
import torch
from qgen_iqp import (suff_stats, IQPBorn, IsingEBM, ARLogistic, MixBernoulli)

torch.set_default_dtype(torch.float64)


def make_target(kind, n, S, P, scale, seed):
    g = torch.Generator().manual_seed(seed)
    a = scale * torch.randn(n, generator=g)
    b = scale * torch.randn(P.shape[1], generator=g)
    m = IQPBorn(n, S, P) if kind == "iqp" else IsingEBM(n, S, P)
    with torch.no_grad():
        m.a.copy_(a); m.b.copy_(b)
        p = torch.exp(m.log_probs())
    return (p / p.sum()).detach()


def entropy(p):
    return -(p * torch.log(p + 1e-300)).sum().item()


def fit_kl(model, p_target, epochs, lrs):
    """Minimize KL(p_target || p_model) exactly; return best excess-NLL (=KL)."""
    H = entropy(p_target)
    best = float("inf")
    for lr in lrs:
        for pm in model.parameters():
            torch.nn.init.normal_(pm, std=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            lp = model.log_probs()
            nll = -(p_target * lp).sum()
            nll.backward()
            opt.step()
        with torch.no_grad():
            nll = -(p_target * model.log_probs()).sum().item()
        best = min(best, nll - H)            # KL >= 0
    return max(best, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    n = args.n_bits
    S, P = suff_stats(n)
    bits01 = (1.0 - S) / 2.0
    npar = n + P.shape[1]
    Kmob = max(2, round((npar + 1) / (n + 1)))
    LRS = [0.05, 0.2, 0.5]
    print(f"[INFO] n_bits={n} (N={2**n}) matched params≈{npar}  "
          f"epochs={args.epochs} lrs={LRS}", flush=True)

    def models():
        return {"iqp": IQPBorn(n, S, P), "ising": IsingEBM(n, S, P),
                "ar": ARLogistic(n, bits01), "mob": MixBernoulli(n, bits01, Kmob)}

    for tgt in ["iqp", "ising"]:
        print(f"\n{'='*64}\n  TARGET = {tgt}-native distribution\n{'='*64}")
        print(f"  {'scale':>6} | {'H(p)':>6} | " +
              " | ".join(f"{k:>8}" for k in ["iqp", "ising", "ar", "mob"]) +
              " | winner")
        for scale in [0.5, 1.0, 2.0, 4.0]:
            p_t = make_target(tgt, n, S, P, scale, seed=123)
            H = entropy(p_t)
            kls = {}
            for name, m in models().items():
                kls[name] = fit_kl(m, p_t, args.epochs, LRS)
            winner = min(kls, key=kls.get)
            print(f"  {scale:>6} | {H:>6.3f} | " +
                  " | ".join(f"{kls[k]:8.4f}" for k in ["iqp", "ising", "ar", "mob"]) +
                  f" | {winner}", flush=True)

    print("\n[VERDICT]")
    print("  If IQP wins (KL≈0) on the iqp target but loses on the ising target,")
    print("  the quantum model is correctly built & trainable — so the ties on")
    print("  NATURAL lensing latents (qgen_iqp: ising/mob win) are a real property")
    print("  of the data, not a broken quantum model.")
    print("[DONE] qpos_control")


if __name__ == "__main__":
    main()
