#!/usr/bin/env python
"""Few-shot data-scaling: classical-only vs +sham-residual vs +quantum-equiv-residual.

For each N-per-class, subsample the train set (seeded), train each mode fresh,
evaluate on the FULL validation set. Quantum's only theoretical edge is the
small-N regime, so this is the decisive test of design ①+③.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

from quantum_equiv import EquivResidualHead, K_LATENT, N_GROUP, N_LAYERS


def subsample(x, y, n_per_class, C, rng):
    idx = []
    for c in range(C):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, size=min(n_per_class, len(ci)), replace=False))
    rng.shuffle(idx)
    return x[idx], y[idx]


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            logits.append(model(x[i:i + bs].to(device)).cpu())
    probs = torch.softmax(torch.cat(logits), 1).numpy()
    yb = label_binarize(y, classes=np.arange(C))
    return (roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
            accuracy_score(y, probs.argmax(1)))


def run_one(mode, xtr, ytr, vx, vy, device, C, epochs, lr):
    torch.manual_seed(0)
    model = EquivResidualHead(n_classes=C, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    xt = torch.from_numpy(xtr).float(); yt = torch.from_numpy(ytr).long()
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(128, len(xt)), shuffle=True)
    best = 0.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        auc, _ = evaluate(model, vx, vy, device, C)
        best = max(best, auc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="img16.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--Ns", default="25,50,100,250,500")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.data), allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx, ty = d["train_x"], d["train_y"]
    vx = torch.from_numpy(d["val_x"]).float(); vy = d["val_y"]
    Ns = [int(s) for s in args.Ns.split(",")]
    print(f"[INFO] few-shot seed={args.seed} K={K_LATENT} group=C{N_GROUP} layers={N_LAYERS} "
          f"epochs={args.epochs} | val={len(vy)} classes={cn}")
    print(f"{'N/class':>8} | {'classical':>10} | {'sham':>10} | {'quantum':>10}")
    print("-" * 48)
    for N in Ns:
        rng = np.random.default_rng(1000 + args.seed)
        xtr, ytr = subsample(tx, ty, N, C, rng)
        res = {}
        for mode in ("none", "sham", "quantum"):
            res[mode] = run_one(mode, xtr, ytr, vx, vy, device, C, args.epochs, args.lr)
        print(f"{N:>8} | {res['none']:>10.4f} | {res['sham']:>10.4f} | {res['quantum']:>10.4f}",
              flush=True)
    print("\n[DONE] higher = better; quantum advantage (if any) shows at small N as quantum > sham")


if __name__ == "__main__":
    main()
