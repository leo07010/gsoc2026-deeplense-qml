#!/usr/bin/env python
"""Pretrain DeepLense-QMAE: masked-image → reconstruct original, maximize fidelity.
Reports mean reconstruction fidelity on val each epoch."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from quantum_mae import QuantumMAE, _BACKEND, K_LATENT, N_Q


@torch.no_grad()
def mean_fid(model, x, device, bs=64):
    model.eval()
    fids = []
    for i in range(0, len(x), bs):
        fids.append(model.reconstruct_fidelity(x[i:i + bs].to(device)).cpu())
    return float(torch.cat(fids).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="img16.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.data), allow_pickle=True)
    tx = torch.from_numpy(d["train_x"]).float()
    vx = torch.from_numpy(d["val_x"]).float()
    print(f"[INFO] {_BACKEND}  train{tuple(tx.shape)} val{tuple(vx.shape)} device={device}")

    if args.smoke:
        tx, vx = tx[:128], vx[:128]; args.epochs = 2
        print("[SMOKE] train=128 val=128, 2 epochs")

    model = QuantumMAE().to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[INFO] params={nparam} (encoder weights + mask token), masked patches={model.mask_patches}")
    print(f"[INIT] val recon fidelity = {mean_fid(model, vx, device):.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(tx), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = model(xb).mean()
            loss.backward(); opt.step()
            run += loss.item() * len(xb)
        f = mean_fid(model, vx, device); best = max(best, f)
        print(f"[ep {ep+1:02d}] train_loss={run/len(tx):.4f}  val_recon_fidelity={f:.4f}", flush=True)

    print(f"\n[DONE] best val reconstruction fidelity = {best:.4f}")


if __name__ == "__main__":
    main()
