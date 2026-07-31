#!/usr/bin/env python
"""Plain classical CNN baseline for the QVF parameter-efficiency claim.

Same CNN encoder + protocol as train_qvf_scratch.py, but the NAE+quantum/sham
readout is replaced by a conventional classical head. This isolates whether
the compact model's accuracy comes from the CNN alone or requires the
NAE+circuit machinery.

Heads:
  --head linear : CNN(128) -> LayerNorm -> Linear(128->3)   (~83k params)
  --head mlp    : CNN(128) -> Linear(128->H) -> ReLU
                             -> LayerNorm -> Linear(H->3)    (H=371 -> ~142.8k,
                             matched to QVF-quantum's 142,795)

Everything else (encoder architecture, optimizer AdamW lr=1e-3 wd=1e-4,
cosine schedule, label_smoothing=0.05, batch=128, seed=42, 9:1 split,
subsample RNG seed 1000+seed, best-val-AUC checkpoint) is copied verbatim
from train_qvf_scratch.py so the comparison is apples-to-apples.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize


class CNNEncoder(nn.Module):
    """Identical to train_qvf_scratch.CNNEncoder (83,072 params at out_dim=128)."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class PlainCNN(nn.Module):
    def __init__(self, n_classes=3, head="linear", feat=128, hid=371):
        super().__init__()
        self.cnn = CNNEncoder(feat)
        if head == "linear":
            self.head = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, n_classes))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat, hid), nn.ReLU(),
                nn.LayerNorm(hid), nn.Linear(hid, n_classes))
        else:
            raise ValueError(head)

    def forward(self, x):
        return self.head(self.cnn(x))


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    probs = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                auc_per=roc_auc_score(yb, probs, average=None, multi_class="ovr"),
                acc=accuracy_score(yy, probs.argmax(1)),
                f1=f1_score(yy, probs.argmax(1), average="macro"),
                cm=confusion_matrix(yy, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    print(f"[INFO] PLAIN-CNN head={args.head} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)} classes={cn}", flush=True)
    if getattr(args, "n_per_class", 0) > 0:                # SAME subsample as QVF-scratch
        rng = np.random.default_rng(1000 + args.seed); idx = []
        for c in range(C):
            ci = np.where(ty.numpy() == c)[0]
            idx.extend(rng.choice(ci, min(args.n_per_class, len(ci)), replace=False))
        rng.shuffle(idx); tx, ty = tx[idx], ty[idx]
        print(f"[INFO] subsampled to N={args.n_per_class}/class → train{tuple(tx.shape)}", flush=True)
    if args.smoke:
        tx, ty, vx, vy = tx[:256], ty[:256], vx[:512], vy[:512]; args.epochs = 2
    model = PlainCNN(C, head=args.head).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={nparam}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    base = os.path.splitext(os.path.basename(args.data))[0]
    best = 0.0; best_per = None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        m = evaluate(model, vx, vy, device, C)
        if m["auc"] > best:
            best = m["auc"]; best_per = m["auc_per"].copy()
        print(f"[ep {ep+1:02d}] val AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f}", flush=True)
    per_s = " ".join(f"{cn[i]}={best_per[i]:.4f}" for i in range(len(cn)))
    print(f"\n[DONE-PLAIN-CNN] head={args.head} {base} N={args.n_per_class or 'full'} "
          f"best_AUC={best:.4f} per_class=({per_s}) params={nparam}", flush=True)


if __name__ == "__main__":
    main()
