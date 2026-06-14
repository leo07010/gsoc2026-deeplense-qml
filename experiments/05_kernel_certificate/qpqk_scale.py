#!/usr/bin/env python
"""
PROJECTED QUANTUM KERNEL (Huang et al. 2021) — the principled remedy for the
exponential concentration that killed the fidelity kernel (measured: g shrinks
with qubit count). PQK uses LOCAL 1-qubit reduced density matrices instead of
global state overlap, so it is immune to that concentration:

  feature(x) = [⟨X_k⟩, ⟨Y_k⟩, ⟨Z_k⟩  for k in 0..n-1]   (3n classical numbers,
               nonlinear functions of x via the n-qubit IQP state)
  K_PQK(x,x') = exp(-γ ||feature(x) - feature(x')||²)

Reported per (n, scale): geometric difference g(K_classical, K_PQK) and
few-shot SVM AUC, PQK vs best classical RBF (on raw PCA features). Same
infrastructure / exact statevectors / convex SVM as qkernel_scale.py.
"""
import os, argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from qkernel_scale import fwht, iqp_states, geom_diff

torch.set_default_dtype(torch.float64)


def pqk_features(V, n, dev):
    """V (B,2^n) statevector → (B, 3n) local Pauli expectations."""
    N = 2 ** n
    p = (V.real ** 2 + V.imag ** 2)                       # (B,2^n)
    idx = torch.arange(N, device=dev)
    bit = ((idx[:, None] >> torch.arange(n, device=dev)[None, :]) & 1).double()  # (2^n,n)
    feats = []
    for k in range(n):
        flip = idx ^ (1 << k)
        f = V.conj() * V[:, flip]                          # (B,2^n)
        xk = f.real.sum(1)
        yk = ((1.0 - 2.0 * bit[:, k]) * f.imag).sum(1)
        zk = ((1.0 - 2.0 * bit[:, k]) * p).sum(1)
        feats += [xk, yk, zk]
    return torch.stack(feats, 1)                           # (B,3n)


def gauss_kernel(A, B, gamma):
    D2 = torch.cdist(A, B) ** 2
    return torch.exp(-gamma * D2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="ssl_features_I.npz")
    ap.add_argument("--n_geom", type=int, default=800)
    ap.add_argument("--qubits", default="8,12,16")
    ap.add_argument("--scales", default="0.5,1.0,2.0")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    Xtr, ytr = d["train_feats"].astype(np.float64), d["train_labels"]
    Xva, yva = d["val_feats"].astype(np.float64), d["val_labels"]
    vi = rng.choice(len(Xva), 1500, replace=False); Xva, yva = Xva[vi], yva[vi]
    yb_va = label_binarize(yva, classes=np.arange(C))
    print(f"[INFO] PQK  device={dev}  qubits={args.qubits}  scales={args.scales}  "
          f"n_geom={args.n_geom}  threshold g>2sqrt(n)={2*np.sqrt(args.n_geom):.1f}",
          flush=True)

    def pqk_med_gamma(F):                                   # median heuristic
        D2 = torch.cdist(F, F) ** 2
        return 1.0 / (D2.median() + 1e-9)

    for n in [int(x) for x in args.qubits.split(",")]:
        pca = PCA(n_components=n, random_state=0).fit(Xtr)
        sc = StandardScaler().fit(pca.transform(Xtr))
        def enc(X, s):
            return torch.from_numpy(np.tanh(sc.transform(pca.transform(X))) * s).to(dev)
        gi = rng.choice(len(Xtr), args.n_geom, replace=False)
        for s in [float(x) for x in args.scales.split(",")]:
            Eg = enc(Xtr[gi], s); Fg = pqk_features(iqp_states(Eg, n, dev), n, dev)
            Fg = (Fg - Fg.mean(0)) / (Fg.std(0) + 1e-9)
            gam = pqk_med_gamma(Fg)
            Kpqk = gauss_kernel(Fg, Fg, gam)
            g_lin = geom_diff(Eg @ Eg.T, Kpqk)
            g_best = g_lin
            for gamma in [0.1, 0.5, 1.0, 5.0]:
                g_best = min(g_best, geom_diff(gauss_kernel(Eg, Eg, gamma), Kpqk))
            tag = "<-- ADVANTAGE POSSIBLE" if g_best > 2 * np.sqrt(args.n_geom) else "(g small)"
            print(f"\n[n={n:2d} scale={s}] PQK g(lin)={g_lin:.2f}  g_min={g_best:.2f}  {tag}",
                  flush=True)
            Fva = pqk_features(iqp_states(enc(Xva, s), n, dev), n, dev)
            Fva = (Fva - Fva.mean(0)) / (Fva.std(0) + 1e-9)
            Evn = enc(Xva, s).cpu().numpy()
            for N in [25, 100, 250]:
                ap_, ar = [], []
                for seed in range(3):
                    r = np.random.default_rng(100 + seed)
                    idx = np.concatenate([r.choice(np.where(ytr == c)[0], N, replace=False)
                                          for c in range(C)])
                    ys = ytr[idx]
                    Fs = pqk_features(iqp_states(enc(Xtr[idx], s), n, dev), n, dev)
                    Fs = (Fs - Fs.mean(0)) / (Fs.std(0) + 1e-9)
                    gam_s = pqk_med_gamma(Fs)
                    Ktr = gauss_kernel(Fs, Fs, gam_s).cpu().numpy()
                    Kva = gauss_kernel(Fva, Fs, gam_s).cpu().numpy()
                    sv = SVC(kernel="precomputed", C=10, probability=True,
                             random_state=0).fit(Ktr, ys)
                    ap_.append(roc_auc_score(yb_va, sv.predict_proba(Kva),
                                             average="macro", multi_class="ovr"))
                    Esn = enc(Xtr[idx], s).cpu().numpy(); best = 0
                    for gamma in [0.1, 0.5, 1.0]:
                        svc = SVC(kernel="rbf", gamma=gamma, C=10, probability=True,
                                  random_state=0).fit(Esn, ys)
                        best = max(best, roc_auc_score(yb_va, svc.predict_proba(Evn),
                                                       average="macro", multi_class="ovr"))
                    ar.append(best)
                win = "PQK WINS" if np.mean(ap_) > np.mean(ar) else ""
                print(f"    N={N:>4}: PQK={np.mean(ap_):.4f}±{np.std(ap_):.4f}  "
                      f"RBF={np.mean(ar):.4f}±{np.std(ar):.4f}  {win}", flush=True)
    print("\n[DONE] qpqk_scale")


if __name__ == "__main__":
    main()
