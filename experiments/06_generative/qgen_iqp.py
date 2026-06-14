#!/usr/bin/env python
"""
IQP generative model on class-conditional lensing latents — the trainable
generative family of Huang et al. (arXiv:2509.09033), evaluated at simulator
scale against PARAMETER-MATCHED classical generative models.

Pipeline:
  SSL features → PCA(n_bits) → sign-vs-median binarization → per-class
  bitstring distributions over {0,1}^n  (n=10/12 → exact enumeration).

Model families (all trained by exact MLE, same optimizer budget):
  iqp     : Born machine  p(x) = |⟨x| H^n D_θ H^n |0⟩|²,
            phases φ(z) = Σ a_i s_i + Σ b_ij s_i s_j   (s=±1)   [n + C(n,2) params]
  ising   : fully-visible Boltzmann machine with the SAME sufficient
            statistics  p(z) ∝ exp(Σ a_i s_i + Σ b_ij s_i s_j)  [same params]
            → the exact classical counterpart: identical parameterization,
              classical exponential vs quantum interference.
  ar      : autoregressive logistic  p(z_i|z_<i)=σ(linear)      [same params]
  mob     : mixture of Bernoulli, K chosen to match params
  cat     : Laplace-smoothed categorical (4095 params)          [upper bound]

Metrics: held-out NLL (primary, proper score), TV to held-out histogram
(secondary; train↔val empirical TV reported as noise floor).

PRE-REGISTERED WIN: iqp val-NLL < ALL matched classical families (ising, ar,
mob) on >=2 of 3 classes AND on the class-mean. Otherwise: honest tie/loss.
"""
import os, argparse
import numpy as np
import torch
from sklearn.decomposition import PCA

torch.set_default_dtype(torch.float64)


# ───────────────────────── utilities ─────────────────────────
def fwht_torch(a):
    """Differentiable unnormalized Walsh–Hadamard transform along last dim."""
    h = 1
    n = a.shape[-1]
    while h < n:
        a = a.reshape(*a.shape[:-1], -1, h * 2)
        x, y = a[..., :h], a[..., h:]
        a = torch.cat([x + y, x - y], dim=-1)
        a = a.reshape(*a.shape[:-2], -1)
        h *= 2
    return a


def suff_stats(n_bits):
    """S (N,n) singles and P (N,nC2) pairs in ±1 convention, N=2^n."""
    N = 2 ** n_bits
    bits = ((np.arange(N)[:, None] >> np.arange(n_bits)[None, :]) & 1)
    s = 1.0 - 2.0 * bits
    pairs = [(i, j) for i in range(n_bits) for j in range(i + 1, n_bits)]
    p = np.stack([s[:, i] * s[:, j] for i, j in pairs], 1)
    return torch.from_numpy(s), torch.from_numpy(p)


# ───────────────────────── model families ─────────────────────────
class IQPBorn(torch.nn.Module):
    def __init__(self, n_bits, S, P):
        super().__init__()
        self.S, self.P = S, P
        self.a = torch.nn.Parameter(0.1 * torch.randn(n_bits))
        self.b = torch.nn.Parameter(0.1 * torch.randn(P.shape[1]))
        self.N = 2 ** n_bits

    def log_probs(self):
        phi = self.S @ self.a + self.P @ self.b            # (N,)
        amp = torch.exp(1j * phi) / np.sqrt(self.N)
        psi = fwht_torch(amp) / np.sqrt(self.N)
        p = (psi.real ** 2 + psi.imag ** 2)
        return torch.log(p + 1e-300)


class IsingEBM(torch.nn.Module):
    """Same sufficient statistics as IQP — the exact classical counterpart."""

    def __init__(self, n_bits, S, P):
        super().__init__()
        self.S, self.P = S, P
        self.a = torch.nn.Parameter(0.1 * torch.randn(n_bits))
        self.b = torch.nn.Parameter(0.1 * torch.randn(P.shape[1]))

    def log_probs(self):
        E = self.S @ self.a + self.P @ self.b
        return E - torch.logsumexp(E, 0)


class ARLogistic(torch.nn.Module):
    """p(z_i | z_<i) = sigmoid(linear); n + nC2 params, enumerated exactly."""

    def __init__(self, n_bits, bits01):
        super().__init__()
        self.n = n_bits
        self.bits = bits01                                  # (N,n) float 0/1
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(0.1 * torch.randn(i + 1)) for i in range(n_bits)])

    def log_probs(self):
        lp = torch.zeros(self.bits.shape[0])
        for i in range(self.n):
            ctx = torch.cat([torch.ones(self.bits.shape[0], 1),
                             self.bits[:, :i]], 1)          # (N, i+1)
            logit = ctx @ self.w[i]
            z = self.bits[:, i]
            lp = lp + z * torch.nn.functional.logsigmoid(logit) \
                    + (1 - z) * torch.nn.functional.logsigmoid(-logit)
        return lp


class MixBernoulli(torch.nn.Module):
    def __init__(self, n_bits, bits01, K):
        super().__init__()
        self.bits = bits01
        self.logit_pi = torch.nn.Parameter(torch.zeros(K))
        self.logit_mu = torch.nn.Parameter(0.1 * torch.randn(K, n_bits))

    def log_probs(self):
        mu_lp1 = torch.nn.functional.logsigmoid(self.logit_mu)     # (K,n)
        mu_lp0 = torch.nn.functional.logsigmoid(-self.logit_mu)
        comp = self.bits @ mu_lp1.T + (1 - self.bits) @ mu_lp0.T   # (N,K)
        logpi = torch.log_softmax(self.logit_pi, 0)
        return torch.logsumexp(comp + logpi, 1)


# ───────────────────────── training / evaluation ─────────────────────────
def train_mle(model, counts_fit, counts_cal, epochs, lrs, tag):
    """Exact MLE: minimize -Σ p̂_fit(x) log p_θ(x). Select lr/epoch by calib NLL."""
    p_fit = counts_fit / counts_fit.sum()
    p_cal = counts_cal / counts_cal.sum()
    best_nll, best_state = float("inf"), None
    for lr in lrs:
        for pmod in model.parameters():
            torch.nn.init.normal_(pmod, std=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            lp = model.log_probs()
            loss = -(p_fit * lp).sum()
            loss.backward()
            opt.step()
            if (ep + 1) % 50 == 0:
                with torch.no_grad():
                    cal = -(p_cal * model.log_probs()).sum().item()
                if cal < best_nll:
                    best_nll = cal
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def metrics(model_lp, counts_val):
    p_val = counts_val / counts_val.sum()
    nll = -(p_val * model_lp).sum().item()                  # nats/sample
    p_mod = torch.exp(model_lp)
    tv = 0.5 * (p_mod - p_val).abs().sum().item()
    return nll, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="ssl_features_I.npz")
    ap.add_argument("--n_bits", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]
    Xtr, ytr = d["train_feats"].astype(np.float64), d["train_labels"]
    Xva, yva = d["val_feats"].astype(np.float64), d["val_labels"]

    n = args.n_bits; N = 2 ** n
    pca = PCA(n_components=n, random_state=0).fit(Xtr)
    Ztr, Zva = pca.transform(Xtr), pca.transform(Xva)
    med = np.median(Ztr, axis=0)
    btr = (Ztr > med).astype(np.int64); bva = (Zva > med).astype(np.int64)
    pow2 = (2 ** np.arange(n)).astype(np.int64)
    itr, iva = btr @ pow2, bva @ pow2

    S, P = suff_stats(n)
    bits01 = (1.0 - S) / 2.0
    n_params_iqp = n + P.shape[1]
    K_mob = max(2, round((n_params_iqp + 1) / (n + 1)))
    print(f"[INFO] n_bits={n} (N={N})  params: iqp/ising/ar={n_params_iqp}  "
          f"mob(K={K_mob})={K_mob*(n+1)-1}  cat={N-1}", flush=True)
    print(f"[INFO] features={args.features}  classes={cn}", flush=True)

    LRS = [0.05, 0.2]
    results = {}
    for c, cname in enumerate(cn):
        ic_all = itr[ytr == c]
        rng = np.random.default_rng(1000 + args.seed)
        perm = rng.permutation(len(ic_all))
        ncal = len(ic_all) // 10
        ic_cal, ic_fit = ic_all[perm[:ncal]], ic_all[perm[ncal:]]
        iv = iva[yva == c]
        cf = torch.bincount(torch.from_numpy(ic_fit), minlength=N).double()
        cc = torch.bincount(torch.from_numpy(ic_cal), minlength=N).double()
        cv = torch.bincount(torch.from_numpy(iv), minlength=N).double()
        # noise floor: empirical fit vs val TV
        tv_floor = 0.5 * (cf / cf.sum() - cv / cv.sum()).abs().sum().item()
        print(f"\n──── class {cname}: fit={len(ic_fit)} cal={len(ic_cal)} "
              f"val={len(iv)}  TV-floor(fit↔val)={tv_floor:.4f} ────", flush=True)

        models = {
            "iqp":   IQPBorn(n, S, P),
            "ising": IsingEBM(n, S, P),
            "ar":    ARLogistic(n, bits01),
            "mob":   MixBernoulli(n, bits01, K_mob),
        }
        for name, m in models.items():
            m = train_mle(m, cf, cc, args.epochs, LRS, f"{cname}/{name}")
            nll, tv = metrics(m.log_probs(), cv)
            npar = sum(p.numel() for p in m.parameters())
            results[(cname, name)] = (nll, tv, npar)
            print(f"  {name:>6}: val NLL={nll:.4f} nats  TV={tv:.4f}  params={npar}",
                  flush=True)
        # categorical upper bound (Laplace smoothing α=0.5)
        p_cat = (cf + 0.5) / (cf + 0.5).sum()
        nll, tv = metrics(torch.log(p_cat), cv)
        results[(cname, "cat")] = (nll, tv, N - 1)
        print(f"  {'cat':>6}: val NLL={nll:.4f} nats  TV={tv:.4f}  params={N-1}",
              flush=True)

    print("\n" + "=" * 68)
    print(f"  SUMMARY (val NLL, nats/sample; lower=better)  n_bits={n}")
    print("=" * 68)
    fams = ["iqp", "ising", "ar", "mob", "cat"]
    print(f"  {'class':>8} | " + " | ".join(f"{f:>8}" for f in fams))
    means = {f: [] for f in fams}
    for cname in cn:
        row = []
        for f in fams:
            v = results[(cname, f)][0]; means[f].append(v); row.append(f"{v:8.4f}")
        print(f"  {cname:>8} | " + " | ".join(row))
    print(f"  {'MEAN':>8} | " + " | ".join(f"{np.mean(means[f]):8.4f}" for f in fams))
    iqp_m = np.mean(means["iqp"])
    best_cl = min(np.mean(means[f]) for f in ["ising", "ar", "mob"])
    wins = sum(1 for cname in cn
               if results[(cname, "iqp")][0] < min(results[(cname, f)][0]
                                                   for f in ["ising", "ar", "mob"]))
    print(f"\n  IQP mean={iqp_m:.4f} vs best matched classical mean={best_cl:.4f} "
          f"| per-class wins vs all matched: {wins}/3")
    print(f"  PRE-REGISTERED WIN = wins>=2 AND iqp mean < classical mean → "
          f"{'WIN' if (wins >= 2 and iqp_m < best_cl) else 'NO WIN'}")
    print("[DONE] qgen_iqp")


if __name__ == "__main__":
    main()
