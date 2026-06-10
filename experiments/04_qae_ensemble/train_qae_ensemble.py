#!/usr/bin/env python
"""
Class-conditional QAE ensemble on leakage-free SSL features (ssl_features_I.npz).

Train ONE one-class autoencoder PER CLASS (no_sub / cdm / axion), each seeing
only its own class's features. At test time every sample gets a z-scored
anomaly score from each model:

    z_c(x) = (score_c(x) − μ_c) / σ_c        score = 1 − fidelity (or distance)

  • closed-set 3-class : argmin_c z_c  (acc / macro-F1 / macro-AUC)
  • substructure anomaly: AUROC(no_sub vs rest) using z_no_sub
  • OPEN-SET (leave-one-class-out): unknown class h, knowns = other two;
    unknown score = min over the two known-class z's; AUROC(unknown vs known).
    Uses the SAME trained models (each never saw other classes) — 3 rotations.

Arms (--arm):
  quantum : 8-qubit trash-qubit QAE (amplitude embed → U(θ) → SWAP reset → U†),
            72 circuit params/class (reuses quantum_mae._recon)
  sham    : classical Linear(256→K→256) AE, same fidelity metric, ~2308 params/class
            (dimension-matched, NOT param-matched — the original control)
  matched : PARAM-MATCHED classical AE: fixed random orthogonal projection
            256→8 (untrainable, analogous to amplitude embedding) + trainable
            Linear(8→K→8), 76 params/class — the honest efficiency test
  maha    : Mahalanobis distance per class on raw 192-d features (Ledoit-Wolf),
            deterministic non-learned baseline

Model selection: best epoch by mean anomaly score on a held-out 10% calibration
split of the SAME class (label-free within the one-class setting). μ_c, σ_c are
computed on that calibration split.

Results append to qae_ensemble_results.csv. --seeds 0,1,2 runs multiple seeds.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.covariance import LedoitWolf

N_PAD = 256                                   # 2^8 amplitudes
PROJ_DIM = 8                                  # matched arm: fixed projection dim


def pad256(x):
    return np.pad(x.astype(np.float32), ((0, 0), (0, N_PAD - x.shape[1])))


class QAE(nn.Module):
    """Trash-qubit quantum AE on 256-d (padded) feature vectors."""

    def __init__(self, recon_fn, enc_shape):
        super().__init__()
        self._recon = recon_fn
        self.w = nn.Parameter(0.1 * torch.randn(enc_shape))

    def score(self, v):                       # v (B,256) → anomaly score (B,)
        vn = v / (v.norm(dim=1, keepdim=True) + 1e-9)
        rho = self._recon(vn, self.w)
        vc = vn.to(rho.dtype)
        t = torch.einsum("bij,bj->bi", rho, vc)
        fid = torch.einsum("bi,bi->b", vc.conj(), t).real.to(v.dtype)
        return 1.0 - fid


class ShamAE(nn.Module):
    """Dimension-matched classical AE (256→K→256), same overlap² fidelity."""

    def __init__(self, k):
        super().__init__()
        self.e = nn.Linear(N_PAD, k)
        self.d = nn.Linear(k, N_PAD)

    def score(self, v):
        vn = v / (v.norm(dim=1, keepdim=True) + 1e-9)
        r = self.d(torch.tanh(self.e(vn)))
        rn = r / (r.norm(dim=1, keepdim=True) + 1e-9)
        return 1.0 - (rn * vn).sum(1) ** 2


class MatchedAE(nn.Module):
    """Param-matched classical AE: FIXED random orthogonal projection 256→8
    (buffer, untrainable — analogous to the fixed amplitude embedding), then
    trainable Linear(8→K→8). Trainable params: 8K+K + 8K+8 = 76 for K=4."""

    def __init__(self, k, gen):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(N_PAD, PROJ_DIM, generator=gen))
        self.register_buffer("P", q)          # (256,8) orthonormal columns
        self.e = nn.Linear(PROJ_DIM, k)
        self.d = nn.Linear(k, PROJ_DIM)

    def score(self, v):
        p = v @ self.P
        pn = p / (p.norm(dim=1, keepdim=True) + 1e-9)
        r = self.d(torch.tanh(self.e(pn)))
        rn = r / (r.norm(dim=1, keepdim=True) + 1e-9)
        return 1.0 - (rn * pn).sum(1) ** 2


class MahaModel:
    """Per-class Mahalanobis on raw 192-d features (Ledoit-Wolf shrinkage)."""

    def __init__(self):
        self.lw = None

    def fit(self, x):                          # x np (n,192)
        self.lw = LedoitWolf().fit(x)

    def score_np(self, x):
        return self.lw.mahalanobis(x)


def train_one_class(model, fit_x, calib_x, device, epochs, lr, bs, tag):
    """Train one-class AE; keep best weights by calib mean anomaly score."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(fit_x), batch_size=bs, shuffle=True)
    calib_x = calib_x.to(device)
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        run = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = model.score(xb).mean()
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            c = batched_score(model, calib_x, device).mean().item()
        if c < best:
            best, best_state = c, copy.deepcopy(model.state_dict())
        print(f"    [{tag} ep {ep+1:02d}] train={run/len(fit_x):.4f} calib={c:.4f}",
              flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def batched_score(model, x, device, bs=256):
    s = []
    for i in range(0, len(x), bs):
        s.append(model.score(x[i:i + bs].to(device)).cpu())
    return torch.cat(s)


def evaluate(z, vy, cn, nosub_idx):
    """z (N,3) z-scored anomaly per class model; lower = more like that class."""
    C = len(cn)
    pred = z.argmin(1)
    probs = torch.softmax(torch.from_numpy(-z), dim=1).numpy()
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(vy, classes=np.arange(C))
    m = dict(
        acc3=accuracy_score(vy, pred),
        f1_3=f1_score(vy, pred, average="macro"),
        auc3=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
        anomaly_auc=roc_auc_score((vy != nosub_idx).astype(int), z[:, nosub_idx]),
    )
    # open-set: leave class h out; unknown score = min z over the other two models
    open_aucs = {}
    for h in range(C):
        known = [c for c in range(C) if c != h]
        minz = z[:, known].min(1)
        is_unknown = (vy == h).astype(int)
        open_aucs[f"open_{cn[h]}"] = roc_auc_score(is_unknown, minz)
    m.update(open_aucs)
    m["open_mean"] = float(np.mean(list(open_aucs.values())))
    return m


def run_seed(arm, seed, d, args, device, recon_fn=None, enc_shape=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cn = [str(c) for c in d["class_names"]]
    C = len(cn)
    nosub = cn.index("no_sub")
    tf_raw, ty = d["train_feats"].astype(np.float32), d["train_labels"]
    vf_raw, vy = d["val_feats"].astype(np.float32), d["val_labels"]
    if args.smoke:
        sub = np.random.default_rng(0).choice(len(vf_raw), 1536, replace=False)
        vf_raw, vy = vf_raw[sub], vy[sub]      # val is class-ordered → sample randomly
    tf = pad256(tf_raw)
    vf = pad256(vf_raw)

    z = np.zeros((len(vf), C), np.float32)
    nparam = 0
    for c in range(C):
        idx = np.where(ty == c)[0]
        rng = np.random.default_rng(1000 + seed)
        rng.shuffle(idx)
        ncal = max(64, len(idx) // 10)
        cal_i, fit_i = idx[:ncal], idx[ncal:]
        if args.smoke:
            fit_i = fit_i[:256]
        tag = f"{arm}/{cn[c]}"

        if arm == "maha":
            mm = MahaModel()
            mm.fit(tf_raw[fit_i])
            cal_s = mm.score_np(tf_raw[cal_i])
            val_s = mm.score_np(vf_raw)
        else:
            if arm == "quantum":
                model = QAE(recon_fn, enc_shape)
            elif arm == "sham":
                model = ShamAE(args.K)
            else:                              # matched
                gen = torch.Generator().manual_seed(7000 + 10 * seed + c)
                model = MatchedAE(args.K, gen)
            nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
            fit_x = torch.from_numpy(tf[fit_i])
            cal_x = torch.from_numpy(tf[cal_i])
            model = train_one_class(model, fit_x, cal_x, device,
                                    args.epochs, args.lr, args.batch_size, tag)
            cal_s = batched_score(model, cal_x, device).numpy()
            val_s = batched_score(model, torch.from_numpy(vf), device).numpy()

        mu, sd = float(cal_s.mean()), float(cal_s.std() + 1e-9)
        z[:, c] = (val_s - mu) / sd
        print(f"  [{tag}] fit={len(fit_i)} calib={len(cal_i)} "
              f"mu={mu:.4f} sd={sd:.4f}", flush=True)

    m = evaluate(z, vy, cn, nosub)
    m.update(arm=arm, seed=seed, K=args.K, params_per_class=nparam)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="ssl_features_I.npz")
    ap.add_argument("--arm", required=True,
                    choices=["quantum", "sham", "matched", "maha"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--out_csv", default="qae_ensemble_results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3

    recon_fn, enc_shape = None, None
    if args.arm == "quantum":
        os.environ["QMAE_K"] = str(args.K)
        from quantum_mae import _recon, enc_shape as es
        recon_fn, enc_shape = _recon, es()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[INFO] QAE-ENSEMBLE arm={args.arm} K={args.K} seeds={seeds} "
          f"features={args.features} smoke={args.smoke}", flush=True)

    cols = ["arm", "seed", "K", "params_per_class", "acc3", "f1_3", "auc3",
            "anomaly_auc", "open_axion", "open_cdm", "open_no_sub", "open_mean"]
    out = os.path.join(repo, args.out_csv)
    rows = []
    for seed in seeds:
        print(f"\n──── arm={args.arm} seed={seed} ────", flush=True)
        m = run_seed(args.arm, seed, d, args, device, recon_fn, enc_shape)
        rows.append(m)
        print("  " + "  ".join(f"{k}={m[k]:.4f}" for k in cols[4:]), flush=True)
        write_header = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(m)

    print("\n" + "=" * 64)
    print(f"  QAE ENSEMBLE SUMMARY  arm={args.arm} K={args.K} "
          f"params/class={rows[0]['params_per_class']}")
    print("=" * 64)
    for k in cols[4:]:
        vals = np.array([r[k] for r in rows])
        print(f"  {k:<12s} mean={vals.mean():.4f}  std={vals.std():.4f}  n={len(vals)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
