#!/usr/bin/env python
"""
Quantum-AE anomaly detection on the 192-d frozen-MAE CLS features (NOT 16x16
pixels). The features come from 64x64 images, so they retain the substructure
signal that downsampling to 16x16 destroyed. Pad 192→256 → amplitude-embed into
8 qubits → quantum AE (U/U† + SWAP trash reset). Train on no_sub only;
reconstruction fidelity = normality score. Binary sub-vs-no_sub ROC-AUC.

  --sham off : quantum AE        --sham on : classical AE (matched K, same metric)
Reference: Alexander 2021 AAE ≈ 0.93 (classical, unsupervised).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cls_features.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ["QMAE_K"] = str(args.K)
    from quantum_mae import _recon, enc_shape, K_LATENT, N_Q

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = list(d["class_names"]); nosub = cn.index("no_sub")
    DIM = 2 ** N_Q                                        # 256

    def pad(x):                                          # (n,192) → (n,256) float32
        return np.pad(x.astype(np.float32), ((0, 0), (0, DIM - x.shape[1])))

    tf = pad(d["train_feats"]); ty = d["train_labels"]
    vf = pad(d["val_feats"]); vy = d["val_labels"]
    tr = torch.from_numpy(tf[ty == nosub])               # train on no_sub only
    vX = torch.from_numpy(vf)
    anom = (vy != nosub).astype(int)
    if args.smoke:
        tr = tr[:256]; args.epochs = 3
    mode = "SHAM-classical" if args.sham else "QUANTUM"
    print(f"[INFO] CLS-feature anomaly {mode} | K={K_LATENT} | train(no_sub)={len(tr)} "
          f"val={len(vX)} anomalies={anom.sum()}")

    class QAE(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
        def fid(self, v):                                # v (B,256)
            vn = v / (v.norm(dim=1, keepdim=True) + 1e-9)
            rho = _recon(vn, self.w)                      # (B,256,256)
            vc = vn.to(rho.dtype)
            t = torch.einsum("bij,bj->bi", rho, vc)
            return torch.einsum("bi,bi->b", vc.conj(), t).real.to(v.dtype)

    class CAE(nn.Module):
        def __init__(self):
            super().__init__(); self.e = nn.Linear(DIM, K_LATENT); self.d = nn.Linear(K_LATENT, DIM)
        def fid(self, v):
            vn = v / (v.norm(dim=1, keepdim=True) + 1e-9)
            r = self.d(torch.tanh(self.e(vn)))
            rn = r / (r.norm(dim=1, keepdim=True) + 1e-9)
            return (rn * vn).sum(1) ** 2

    model = (CAE() if args.sham else QAE()).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[INFO] params={nparam}")

    @torch.no_grad()
    def auc():
        model.eval(); s = []
        for i in range(0, len(vX), 256):
            s.append((1.0 - model.fid(vX[i:i + 256].to(device))).cpu())
        return roc_auc_score(anom, torch.cat(s).numpy())

    print(f"[INIT] anomaly ROC-AUC = {auc():.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(tr), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(); loss = (1.0 - model.fid(xb)).mean(); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        a = auc(); best = max(best, a)
        print(f"[ep {ep+1:02d}] recon_loss={run/len(tr):.4f} anomaly_ROC_AUC={a:.4f}", flush=True)

    print("\n" + "=" * 52)
    print(f"  CLS-FEATURE QUANTUM-AE ANOMALY ({mode}, K={K_LATENT})")
    print("=" * 52)
    print(f"Best anomaly ROC-AUC : {best:.4f}   params={nparam}")
    print(f"(reference: Alexander 2021 AAE ≈ 0.93)")
    print("=" * 52)


if __name__ == "__main__":
    main()
