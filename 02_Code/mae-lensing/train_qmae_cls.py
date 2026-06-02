#!/usr/bin/env python
"""Downstream 3-class on the QMAE latent (full 16x16 image), with sham control.

  --sham off : quantum encoder (8-qubit amplitude embed + U(θ)) → latent ⟨Z⟩
  --sham on  : classical Linear(256→K) bottleneck (matched dim, no circuit)

Trains encoder+head end-to-end on labels. Same protocol for both → the only
difference is whether the K-dim latent is produced by a quantum circuit.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from quantum_mae import QMAEClassifier, K_LATENT, _BACKEND


def evaluate(model, x, y, device, C, bs=128):
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            logits.append(model(x[i:i + bs].to(device)).cpu())
    logits = torch.cat(logits); probs = torch.softmax(logits, 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(yy, probs.argmax(1)),
                f1=f1_score(yy, probs.argmax(1), average="macro"),
                cm=confusion_matrix(yy, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="img16.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.data), allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float(); ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float(); vy = torch.from_numpy(d["val_y"]).long()
    mode = "SHAM-classical" if args.sham else "QUANTUM"
    print(f"[INFO] {_BACKEND} | downstream={mode} | latent_dim K={K_LATENT} | "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)} classes={cn}")

    if args.smoke:
        tx, ty, vx, vy = tx[:256], ty[:256], vx[:256], vy[:256]; args.epochs = 2
        print("[SMOKE] 256/256, 2 epochs")

    model = QMAEClassifier(n_classes=C, sham=args.sham).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params: {nparam}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        m = evaluate(model, vx, vy, device, C); best = max(best, m["auc"])
        ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
        print(f"[ep {ep+1:02d}] loss={run/len(tx):.4f} val AUC={m['auc']:.4f} "
              f"acc={m['acc']:.4f} f1={m['f1']:.4f} axion_rec={axr:.4f}", flush=True)

    m = evaluate(model, vx, vy, device, C)
    ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
    print("\n" + "=" * 56)
    print(f"     QMAE-latent 3-class ({mode}, K={K_LATENT})")
    print("=" * 56)
    print(f"Final AUC : {m['auc']:.4f}  acc {m['acc']:.4f}  f1 {m['f1']:.4f}")
    print(f"Best  AUC : {best:.4f}   axion recall {axr:.4f}")
    print("Confusion (rows=true):")
    print("        " + "  ".join(f"{c:>7}" for c in cn))
    for i, row in enumerate(m["cm"]):
        print(f"{cn[i]:>7} " + "  ".join(f"{v:>7d}" for v in row))
    print("=" * 56)


if __name__ == "__main__":
    main()
