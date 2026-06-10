#!/usr/bin/env python
"""
Quantum-AE anomaly detection on DeepLense (Stretch Goal A; cf. Alexander 2021 AAE AUC≈0.93).

Train an autoencoder on no_sub ONLY (reconstruct, no mask). At test time the
reconstruction fidelity is the "normality" score: substructure (axion/cdm)
reconstructs worse → higher anomaly score. Binary task: sub vs no_sub, ROC-AUC.

  --sham off : quantum autoencoder (8-qubit amplitude embed, U/U†, SWAP trash reset)
  --sham on  : classical AE with matched latent dim K and the same fidelity metric

The point is NOT to beat classical accuracy but to test the documented quantum
value axis: comparable anomaly AUC with far fewer parameters.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="img16.npz")
    ap.add_argument("--K", type=int, default=4, help="latent qubits / dim (tighter = better anomaly sep)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ["QMAE_K"] = str(args.K)                    # before importing quantum_mae
    from quantum_mae import QuantumMAE, ClassicalAE, K_LATENT, enc_shape

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.data), allow_pickle=True)
    cn = list(d["class_names"]); nosub = cn.index("no_sub")
    tx, ty = d["train_x"], d["train_y"]
    vx, vy = d["val_x"], d["val_y"]

    # train ONLY on no_sub (normal class)
    tr_norm = tx[ty == nosub]
    if args.smoke:
        tr_norm = tr_norm[:256]; args.epochs = 3
    tr = torch.from_numpy(tr_norm).float()
    vX = torch.from_numpy(vx).float()
    anom_label = (vy != nosub).astype(int)                # 1 = substructure (anomaly)
    mode = "SHAM-classical" if args.sham else "QUANTUM"
    print(f"[INFO] anomaly-detection {mode} | latent K={K_LATENT} | train(no_sub)={len(tr)} "
          f"val={len(vX)} (anomalies={anom_label.sum()})")

    model = (ClassicalAE() if args.sham else QuantumMAE(mask_patches=())).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[INFO] params={nparam}")

    @torch.no_grad()
    def auc():
        model.eval(); s = []
        for i in range(0, len(vX), 128):
            s.append((1.0 - model.reconstruct_fidelity(vX[i:i + 128].to(device))).cpu())
        score = torch.cat(s).numpy()                      # anomaly score = 1 - fidelity
        return roc_auc_score(anom_label, score)

    print(f"[INIT] anomaly ROC-AUC = {auc():.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(tr), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(); loss = model(xb).mean(); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        a = auc(); best = max(best, a)
        print(f"[ep {ep+1:02d}] recon_loss={run/len(tr):.4f}  anomaly_ROC_AUC={a:.4f}", flush=True)

    print("\n" + "=" * 52)
    print(f"   QUANTUM-AE ANOMALY DETECTION ({mode}, K={K_LATENT})")
    print("=" * 52)
    print(f"Best anomaly ROC-AUC : {best:.4f}   params={nparam}")
    print(f"(reference: Alexander 2021 AAE ≈ 0.93, classical)")
    print("=" * 52)


if __name__ == "__main__":
    main()
