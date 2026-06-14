#!/usr/bin/env python
"""
QUBIT-SCALING of the quantum kernel — the one path where qubit count has real
theoretical leverage (Huang et al.: the geometric difference g can grow with
the number of qubits, and large g is the necessary condition for a quantum-
kernel advantage). Earlier g≈3-6 was measured at 8 qubits ONLY; this scales
to 12 and 16 qubits on GPU (exact statevectors, 2^16 = 65536-d).

For n ∈ {8,12,16}:
  PCA-n of SSL CLS features → bounded encoding → 2-layer IQP feature map
    |φ(x)⟩ = U_Z(x) H^n U_Z(x) H^n |0⟩,  U_Z = exp(i[Σ x_i Z_i + Σ x_i x_j Z_iZ_j])
  computed exactly via batched Walsh–Hadamard transform on GPU.
  K_Q(x,x') = |⟨φ(x)|φ(x')⟩|².

Reports, per (n, encoding-scale):
  • geometric difference g(K_C, K_Q) vs linear + RBF-sweep  (g >> sqrt(N) needed)
  • few-shot SVM AUC: quantum kernel vs best classical kernel, N/class up to 250
"""
import os, argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

torch.set_default_dtype(torch.float64)


def fwht(a):                                          # batched (B, 2^n) complex, GPU
    h, n = 1, a.shape[-1]
    while h < n:
        a = a.reshape(*a.shape[:-1], -1, h * 2)
        x, y = a[..., :h], a[..., h:]
        a = torch.cat([x + y, x - y], -1).reshape(*a.shape[:-2], -1)
        h *= 2
    return a


def iqp_states(X, n, dev):
    """X (N,n) → statevectors (N,2^n) complex, 2-layer IQP."""
    N = 2 ** n
    bits = ((torch.arange(N, device=dev)[:, None] >> torch.arange(n, device=dev)[None, :]) & 1)
    Z = 1.0 - 2.0 * bits.double()                     # (2^n, n) ∈ {+1,-1}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ZZ = torch.stack([Z[:, i] * Z[:, j] for i, j in pairs], 1)   # (2^n, P)
    Xp = torch.stack([X[:, i] * X[:, j] for i, j in pairs], 1)   # (B, P)
    phase = X @ Z.T + Xp @ ZZ.T                       # (B, 2^n)
    v = torch.full((len(X), N), 1.0 / np.sqrt(N), dtype=torch.complex128, device=dev)
    v = v * torch.exp(1j * phase)
    v = fwht(v) / np.sqrt(N)
    v = v * torch.exp(1j * phase)
    return v


def kq(V1, V2=None):
    V2 = V1 if V2 is None else V2
    return (torch.abs(V1 @ V2.conj().T) ** 2).real


def geom_diff(Kc, Kq, lam=1e-6):
    n = len(Kc)
    Kc = Kc * (n / torch.trace(Kc)); Kq = Kq * (n / torch.trace(Kq))
    wq, Uq = torch.linalg.eigh(Kq)
    sq = Uq @ torch.diag(torch.sqrt(torch.clamp(wq, min=0))) @ Uq.T
    M = sq @ torch.linalg.solve(Kc + lam * n * torch.eye(n, dtype=Kc.dtype, device=Kc.device), sq)
    return float(torch.sqrt(torch.clamp(torch.linalg.eigvalsh(M).max(), min=0)))


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
    print(f"[INFO] device={dev} qubits={args.qubits} scales={args.scales} "
          f"n_geom={args.n_geom} sqrt(n_geom)={np.sqrt(args.n_geom):.1f}", flush=True)

    for n in [int(x) for x in args.qubits.split(",")]:
        pca = PCA(n_components=n, random_state=0).fit(Xtr)
        sc = StandardScaler().fit(pca.transform(Xtr))
        def enc(X, s):
            return torch.from_numpy(np.tanh(sc.transform(pca.transform(X))) * s).to(dev)
        gi = rng.choice(len(Xtr), args.n_geom, replace=False)
        for s in [float(x) for x in args.scales.split(",")]:
            Eg = enc(Xtr[gi], s); Vg = iqp_states(Eg, n, dev); Kq = kq(Vg)
            g_lin = geom_diff(Eg @ Eg.T, Kq)
            g_best = g_lin
            for gamma in [0.1, 0.5, 1.0, 5.0]:
                D2 = torch.cdist(Eg, Eg) ** 2
                g_best = min(g_best, geom_diff(torch.exp(-gamma * D2), Kq))
            print(f"\n[n={n:2d} scale={s}] g(lin)={g_lin:.2f}  g_min(vs classical)={g_best:.2f}"
                  f"  {'<-- ADVANTAGE POSSIBLE' if g_best > 2*np.sqrt(args.n_geom) else '(g small)'}",
                  flush=True)
            # few-shot
            Eva = enc(Xva, s); Vva = iqp_states(Eva, n, dev)
            for N in [25, 100, 250]:
                aq, ar = [], []
                for seed in range(3):
                    r = np.random.default_rng(100 + seed)
                    idx = np.concatenate([r.choice(np.where(ytr == c)[0], N, replace=False)
                                          for c in range(C)])
                    Es = enc(Xtr[idx], s); Vs = iqp_states(Es, n, dev); ys = ytr[idx]
                    Ktr = kq(Vs).cpu().numpy(); Kva = kq(Vva, Vs).cpu().numpy()
                    sv = SVC(kernel="precomputed", C=10, probability=True,
                             random_state=0).fit(Ktr, ys)
                    aq.append(roc_auc_score(yb_va, sv.predict_proba(Kva),
                                            average="macro", multi_class="ovr"))
                    Esn, Evn = Es.cpu().numpy(), Eva.cpu().numpy()
                    best = 0
                    for gamma in [0.1, 0.5, 1.0]:
                        svc = SVC(kernel="rbf", gamma=gamma, C=10, probability=True,
                                  random_state=0).fit(Esn, ys)
                        best = max(best, roc_auc_score(yb_va, svc.predict_proba(Evn),
                                                       average="macro", multi_class="ovr"))
                    ar.append(best)
                print(f"    N={N:>4}: quantum={np.mean(aq):.4f}±{np.std(aq):.4f}  "
                      f"RBF={np.mean(ar):.4f}±{np.std(ar):.4f}  "
                      f"{'Q WINS' if np.mean(aq) > np.mean(ar) else ''}", flush=True)
    print("\n[DONE] qkernel_scale")


if __name__ == "__main__":
    main()
