#!/usr/bin/env python
"""
Train the Quantum-Classical Transformer (QCT) head (quantum_fusion_qct.py):
classical ViT patch tokens + quantum readout tokens fused by mixed self-attention.

Needs:
  cls_features.npz   (labels, head weights, class names, CLS feats)  -- from extract_features.py
  train_patches.npy / val_patches.npy  (float16 patch tokens)         -- from extract_patch_features.py

Usage:
    python train_qct.py --epochs 20 [--reupload] [--sham]
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize


class PatchDS(Dataset):
    def __init__(self, patches, cls, labels):
        self.p, self.c, self.y = patches, cls, labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.p[i])).float(),
                torch.from_numpy(np.asarray(self.c[i])).float(),
                int(self.y[i]))


def evaluate(head, loader, device, C):
    head.eval(); logits, ys = [], []
    with torch.no_grad():
        for p, c, y in loader:
            logits.append(head(p.to(device), c.to(device)).cpu()); ys.append(y)
    logits = torch.cat(logits); y = torch.cat(ys).numpy()
    probs = torch.softmax(logits, 1).numpy()
    yb = label_binarize(y, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(y, probs.argmax(1)),
                f1=f1_score(y, probs.argmax(1), average="macro"),
                cm=confusion_matrix(y, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs_qct")
    ap.add_argument("--reupload", action="store_true")
    ap.add_argument("--sham", action="store_true",
                    help="ablation: replace quantum tokens with a classical projection (same shapes)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.reupload:
        os.environ["QF_REUPLOAD"] = "1"
    if args.sham:
        os.environ["QF_SHAM"] = "1"
    import quantum_fusion_qct as M
    QuantumFusionHead, _BACKEND = M.QuantumFusionHead, M._BACKEND

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo, args.out); os.makedirs(out_dir, exist_ok=True)

    d = np.load(os.path.join(repo, "cls_features.npz"), allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tc, vc = d["train_feats"], d["val_feats"]
    ty, vy = d["train_labels"], d["val_labels"]
    tp = np.load(os.path.join(repo, "train_patches.npy"), mmap_mode="r")
    vp = np.load(os.path.join(repo, "val_patches.npy"), mmap_mode="r")
    print(f"[INFO] backend={_BACKEND} sham={args.sham}  patches train{tp.shape} val{vp.shape}  classes={cn}")

    if args.smoke:
        tp, tc, ty = tp[:256], tc[:256], ty[:256]
        vp, vc, vy = vp[:512], vc[:512], vy[:512]
        args.epochs = 1; print("[SMOKE] train=256 val=512 1 epoch")

    tr = DataLoader(PatchDS(tp, tc, ty), batch_size=args.batch_size, shuffle=True, num_workers=4)
    va = DataLoader(PatchDS(vp, vc, vy), batch_size=args.batch_size, shuffle=False, num_workers=4)

    head = QuantumFusionHead(in_dim=tc.shape[1], n_classes=C).to(device)
    nparam = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[INFO] trainable params: {nparam/1e3:.1f}k")
    base = evaluate(head, va, device, C)
    print(f"[INIT] AUC={base['auc']:.4f} acc={base['acc']:.4f} f1={base['f1']:.4f}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0
    for ep in range(args.epochs):
        head.train(); run = 0.0
        for p, c, y in tr:
            p, c, y = p.to(device), c.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(head(p, c), y); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); run += loss.item() * len(y)
        sched.step()
        m = evaluate(head, va, device, C)
        ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
        print(f"[ep {ep+1:02d}] loss={run/len(ty):.4f} val AUC={m['auc']:.4f} "
              f"acc={m['acc']:.4f} f1={m['f1']:.4f} axion_rec={axr:.4f}", flush=True)
        if m["auc"] > best:
            best = m["auc"]
            torch.save(head.state_dict(), os.path.join(out_dir, f"qct_{'sham' if args.sham else 'q'}_best.pth"))

    m = evaluate(head, va, device, C)
    ai = cn.index("axion"); axr = m["cm"][ai, ai] / m["cm"][ai].sum()
    print("\n" + "=" * 60)
    print(f"        QCT RESULTS ({'SHAM-classical' if args.sham else 'QUANTUM'})")
    print("=" * 60)
    print(f"Final AUC : {m['auc']:.4f}  acc {m['acc']:.4f}  f1 {m['f1']:.4f}")
    print(f"Best  AUC : {best:.4f}")
    print(f"axion recall : {axr:.4f}  (baseline 0.754, classical-only 0.788)")
    print("Confusion (rows=true):")
    print("        " + "  ".join(f"{c:>7}" for c in cn))
    for i, row in enumerate(m["cm"]):
        print(f"{cn[i]:>7} " + "  ".join(f"{v:>7d}" for v in row))
    print("=" * 60)


if __name__ == "__main__":
    main()
