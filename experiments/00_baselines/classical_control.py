#!/usr/bin/env python
"""Classical-only control: fine-tune ONLY a linear head (192->3) on the cached
CLS features, initialised from the shipped baseline head. No quantum branch.

This isolates how much of the "fusion" gain is just classical fine-tuning of
the trainable linear head — the essential attribution baseline (cf. the
classical-only variant in arXiv:2512.19180). Same optimizer/epochs as the
quantum runs.
"""
import os, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def evaluate(head, feats, labels, C, device, bs=512):
    head.eval()
    with torch.no_grad():
        logits = torch.cat([head(feats[i:i+bs].to(device)).cpu()
                            for i in range(0, len(feats), bs)])
    probs = torch.softmax(logits, 1).numpy(); y = labels.numpy()
    yb = label_binarize(y, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(y, probs.argmax(1)),
                f1=f1_score(y, probs.argmax(1), average="macro"),
                cm=confusion_matrix(y, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cls_features.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__)).rsplit("/slurm", 1)[0]
    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tf = torch.from_numpy(d["train_feats"]).float(); tl = torch.from_numpy(d["train_labels"]).long()
    vf = torch.from_numpy(d["val_feats"]).float(); vl = torch.from_numpy(d["val_labels"]).long()

    head = nn.Linear(tf.shape[1], C).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(d["head_weight"]))
        head.bias.copy_(torch.as_tensor(d["head_bias"]))
    print(f"[INFO] CLASSICAL-ONLY linear head, init from shipped baseline. classes={cn}")
    base = evaluate(head, vf, vl, C, device)
    print(f"[BASELINE @init] AUC={base['auc']:.4f} acc={base['acc']:.4f} f1={base['f1']:.4f}")

    opt = torch.optim.Adam(head.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tf, tl), batch_size=args.batch_size, shuffle=True)
    best = base["auc"]
    for ep in range(args.epochs):
        head.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(head(x), y).backward(); opt.step()
        m = evaluate(head, vf, vl, C, device); best = max(best, m["auc"])
        print(f"[ep {ep+1:02d}] val AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f}", flush=True)

    m = evaluate(head, vf, vl, C, device)
    ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
    print("\n" + "=" * 56)
    print("        CLASSICAL-ONLY (linear head fine-tune)")
    print("=" * 56)
    print(f"Baseline AUC : {base['auc']:.4f}  acc {base['acc']:.4f}")
    print(f"Final    AUC : {m['auc']:.4f}  acc {m['acc']:.4f}  f1 {m['f1']:.4f}")
    print(f"Best val AUC : {best:.4f}")
    print(f"axion recall : {axr:.4f}  (baseline ~0.754)")
    print("Confusion (rows=true):")
    print("        " + "  ".join(f"{c:>7}" for c in cn))
    for i, row in enumerate(m["cm"]):
        print(f"{cn[i]:>7} " + "  ".join(f"{v:>7d}" for v in row))
    print("=" * 56)


if __name__ == "__main__":
    main()
