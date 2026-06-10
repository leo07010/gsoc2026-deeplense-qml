#!/usr/bin/env python
"""Train the QVF-style classifier on 192-d frozen-MAE CLS features (3-class).
Quantum (learnable neural amplitude encoding + PQC) vs sham (same NAE, classical)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            logits.append(model(x[i:i + bs].to(device)).cpu())
    probs = torch.softmax(torch.cat(logits), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(yy, probs.argmax(1)),
                f1=f1_score(yy, probs.argmax(1), average="macro"),
                cm=confusion_matrix(yy, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cls_features.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=0, help="0 = full data; else few-shot")
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from quantum_qvf import QVFClassifier, _BACKEND, K_LATENT
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tf = d["train_feats"].astype(np.float32); ty = d["train_labels"]
    vx = torch.from_numpy(d["val_feats"].astype(np.float32)); vy = torch.from_numpy(d["val_labels"]).long()

    if args.n_per_class > 0:
        rng = np.random.default_rng(1000 + args.seed); idx = []
        for c in range(C):
            ci = np.where(ty == c)[0]; idx.extend(rng.choice(ci, min(args.n_per_class, len(ci)), replace=False))
        rng.shuffle(idx); tf, ty = tf[idx], ty[idx]
    if args.smoke:
        tf, ty = tf[:256], ty[:256]; args.epochs = 2
    tx = torch.from_numpy(tf); tyt = torch.from_numpy(ty).long()
    mode = "SHAM-classical" if args.sham else "QUANTUM"
    print(f"[INFO] {_BACKEND} | {mode} | train={len(tx)} val={len(vx)} N/class={args.n_per_class or 'full'}")

    model = QVFClassifier(in_dim=tf.shape[1], n_classes=C, sham=args.sham).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={nparam}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tx, tyt), batch_size=min(args.batch_size, len(tx)), shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        m = evaluate(model, vx, vy, device, C); best = max(best, m["auc"])
        ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
        print(f"[ep {ep+1:02d}] val AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} axion_rec={axr:.4f}",
              flush=True)
    m = evaluate(model, vx, vy, device, C)
    ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
    print("\n" + "=" * 52)
    print(f"   QVF 3-class ({mode}, K={K_LATENT})  params={nparam}")
    print("=" * 52)
    print(f"Final AUC {m['auc']:.4f}  acc {m['acc']:.4f}  f1 {m['f1']:.4f} | Best AUC {best:.4f} | axion_rec {axr:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
