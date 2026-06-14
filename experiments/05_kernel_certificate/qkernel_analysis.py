#!/usr/bin/env python
"""
Quantum-kernel advantage test (Huang et al. 2021, "Power of data").

Instead of training yet another hybrid (expectation-value readouts provably
collapse to classical function classes — all our shams tie), we test the ONE
discriminative regime where quantum can differ: the FIDELITY KERNEL of an
IQP feature map (Havlicek 2019), whose Gram matrix is conjectured classically
hard at scale, evaluated in the scarce-data regime.

Outputs:
  1. geometric difference  g(K_C, K_Q) = sqrt(|| sqrt(K_Q) K_C^{-1} sqrt(K_Q) ||)
     vs each classical kernel (linear / RBF-sweep). Small g ⇒ a classical
     kernel can always match the quantum one on ANY labels (no advantage
     possible on this data). Large g ⇒ advantage possible; then check:
  2. few-shot SVM AUC curves: quantum kernel vs best classical kernel,
     N ∈ {10,25,50,100,250}/class × 5 subsample seeds.

Encoding: PCA-8 of the SSL CLS features → IQP map
  |φ(x)⟩ = U_Z(x) H^{⊗8} U_Z(x) H^{⊗8} |0⟩,
  U_Z(x) = diag exp(i[Σ x_i Z_i + Σ_{i<j} x_i x_j Z_i Z_j])
computed exactly with a Walsh–Hadamard transform (8 qubits → 256-d statevector).
K_Q(x,x') = |⟨φ(x)|φ(x')⟩|².
"""
import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler

N_Q = 8
DIM = 2 ** N_Q


def fwht(a):                                       # Walsh-Hadamard along last axis
    h = 1
    a = a.copy()
    while h < a.shape[-1]:
        a = a.reshape(*a.shape[:-1], -1, h * 2)
        x, y = a[..., :h].copy(), a[..., h:].copy()
        a[..., :h], a[..., h:] = x + y, x - y
        a = a.reshape(*a.shape[:-2], -1)
        h *= 2
    return a


# bit patterns of the 256 basis states (z_i = ±1 convention)
_BITS = ((np.arange(DIM)[:, None] >> np.arange(N_Q)[None, :]) & 1)
_Z = 1.0 - 2.0 * _BITS                              # (256, 8) in {+1,-1}
_PAIRS = [(i, j) for i in range(N_Q) for j in range(i + 1, N_Q)]
_ZZ = np.stack([_Z[:, i] * _Z[:, j] for i, j in _PAIRS], 1)   # (256, 28)


def iqp_states(X):
    """X (n,8) scaled features → statevectors (n,256) complex128."""
    phase = X @ _Z.T + np.stack([X[:, i] * X[:, j] for i, j in _PAIRS], 1) @ _ZZ.T
    v = np.full((len(X), DIM), 1.0 / np.sqrt(DIM), dtype=np.complex128)
    v = v * np.exp(1j * phase)
    v = fwht(v) / np.sqrt(DIM)
    v = v * np.exp(1j * phase)
    return v


def gram_quantum(V1, V2=None):
    V2 = V1 if V2 is None else V2
    return np.abs(V1 @ V2.conj().T) ** 2            # fidelity kernel


def geometric_difference(Kc, Kq, lam=1e-6):
    """g = sqrt(|| sqrt(Kq) (Kc+lam)^{-1} sqrt(Kq) ||_inf), kernels trace-normalized."""
    n = len(Kc)
    Kc = Kc * (n / np.trace(Kc)); Kq = Kq * (n / np.trace(Kq))
    wq, Uq = np.linalg.eigh(Kq)
    sq = Uq @ np.diag(np.sqrt(np.clip(wq, 0, None))) @ Uq.T
    M = sq @ np.linalg.solve(Kc + lam * n * np.eye(n), sq)
    return float(np.sqrt(max(np.linalg.eigvalsh(M).max(), 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="ssl_features_I.npz")
    ap.add_argument("--n_geom", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=1.0, help="feature scaling into phases")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    Xtr_full, ytr_full = d["train_feats"].astype(np.float64), d["train_labels"]
    Xva, yva = d["val_feats"].astype(np.float64), d["val_labels"]

    # PCA-8 fitted on train, standardized, then bounded scaling into phases
    pca = PCA(n_components=N_Q, random_state=0).fit(Xtr_full)
    sc = StandardScaler().fit(pca.transform(Xtr_full))
    def enc(X):
        z = sc.transform(pca.transform(X))
        return np.tanh(z) * args.scale              # bounded phases, scale sweepable
    vi = rng.choice(len(Xva), 2000, replace=False)
    Xva, yva = Xva[vi], yva[vi]
    Eva = enc(Xva); Vva = iqp_states(Eva)

    # ── 1) geometric difference on a fixed subsample ──
    gi = rng.choice(len(Xtr_full), args.n_geom, replace=False)
    Eg = enc(Xtr_full[gi]); Vg = iqp_states(Eg)
    Kq = gram_quantum(Vg)
    print(f"[GEOM] n={args.n_geom} scale={args.scale}", flush=True)
    best_g = None
    Klin = Eg @ Eg.T
    g = geometric_difference(Klin, Kq); best_g = g
    print(f"  g(linear , K_Q) = {g:.2f}")
    for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        D2 = ((Eg[:, None, :] - Eg[None, :, :]) ** 2).sum(-1)
        Krbf = np.exp(-gamma * D2)
        g = geometric_difference(Krbf, Kq)
        best_g = min(best_g, g)
        print(f"  g(RBF γ={gamma:<4}, K_Q) = {g:.2f}", flush=True)
    print(f"[GEOM] min over classical kernels: g_min = {best_g:.2f}  "
          f"(g≈1 ⇒ no quantum-kernel advantage possible; "
          f"advantage needs g >> sqrt(n)·ε)", flush=True)

    # ── 2) few-shot SVM: quantum vs best classical ──
    yb_va = label_binarize(yva, classes=np.arange(C))
    print(f"\n[FEWSHOT] SVM, val n={len(yva)}; mean±std over 5 seeds")
    print(f"  {'N/class':>8} | {'quantum':>15} | {'RBF best':>15} | {'linear':>15}")
    for N in [10, 25, 50, 100, 250]:
        aucs = {k: [] for k in ["q", "rbf", "lin"]}
        for seed in range(5):
            r = np.random.default_rng(100 + seed)
            idx = np.concatenate([r.choice(np.where(ytr_full == c)[0], N, replace=False)
                                  for c in range(C)])
            Xs, ys = Xtr_full[idx], ytr_full[idx]
            Es = enc(Xs); Vs = iqp_states(Es)
            # quantum fidelity kernel SVM
            Kq_tr = gram_quantum(Vs); Kq_va = gram_quantum(Vva, Vs)
            sv = SVC(kernel="precomputed", C=10, probability=True,
                     random_state=0).fit(Kq_tr, ys)
            aucs["q"].append(roc_auc_score(yb_va, sv.predict_proba(Kq_va),
                                           average="macro", multi_class="ovr"))
            # classical RBF (per-split best gamma)
            best = 0
            for gamma in [0.05, 0.1, 0.5, 1.0]:
                svc = SVC(kernel="rbf", gamma=gamma, C=10, probability=True,
                          random_state=0).fit(Es, ys)
                a = roc_auc_score(yb_va, svc.predict_proba(Eva),
                                  average="macro", multi_class="ovr")
                best = max(best, a)
            aucs["rbf"].append(best)
            svl = SVC(kernel="linear", C=10, probability=True,
                      random_state=0).fit(Es, ys)
            aucs["lin"].append(roc_auc_score(yb_va, svl.predict_proba(Eva),
                                             average="macro", multi_class="ovr"))
        f = {k: (np.mean(v), np.std(v)) for k, v in aucs.items()}
        print(f"  {N:>8} | {f['q'][0]:.4f}±{f['q'][1]:.4f} | "
              f"{f['rbf'][0]:.4f}±{f['rbf'][1]:.4f} | "
              f"{f['lin'][0]:.4f}±{f['lin'][1]:.4f}", flush=True)
    print("\n[DONE] quantum-kernel advantage test complete")


if __name__ == "__main__":
    main()
