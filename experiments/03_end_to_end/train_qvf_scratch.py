#!/usr/bin/env python
"""
From-SCRATCH QVF on raw 64x64 lensing images.

  image → trainable CNN encoder → feature → NAE (neural amplitude encoding,
  learnable energy → Boltzmann amplitudes) → 8-qubit PQC → ⟨Z⟩ → head → 3-class.
Everything (CNN, NAE, circuit, head) trained TOGETHER from scratch — the full
from-scratch counterpart of the frozen-feature QVF.

  --sham : NAE → classical Linear (matched dim, no circuit) — isolates the
           quantum circuit given the same trainable encoder + learnable encoding.
Data: model_X.npz from cache_model.py.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from quantum_qvf import _circuit, enc_shape, K_LATENT, N_Q, DIM   # DIM=256


class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)              # (B, out_dim)


class NAE(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.energy = nn.Sequential(nn.Linear(in_dim, hid), nn.Tanh(), nn.Linear(hid, DIM))

    def forward(self, x):
        return torch.sqrt(torch.softmax(-self.energy(x), dim=1) + 1e-12)   # amplitudes


class QVFScratch(nn.Module):
    def __init__(self, n_classes=3, sham=False, feat=128):
        super().__init__()
        self.sham = sham
        self.cnn = CNNEncoder(feat)
        self.nae = NAE(feat)
        if sham:
            self.cl = nn.Linear(DIM, K_LATENT)
        else:
            self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
        self.head = nn.Sequential(nn.LayerNorm(K_LATENT), nn.Linear(K_LATENT, n_classes))

    def forward(self, x):
        amp = self.nae(self.cnn(x))
        if self.sham:
            z = torch.tanh(self.cl(amp))
        else:
            z = torch.stack(_circuit(amp, self.w), dim=-1).to(amp.dtype)
        return self.head(z)


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    probs = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(yy, probs.argmax(1)),
                f1=f1_score(yy, probs.argmax(1), average="macro"),
                cm=confusion_matrix(yy, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    mode = "SHAM" if args.sham else "QUANTUM"
    print(f"[INFO] FROM-SCRATCH QVF {mode} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)} classes={cn}", flush=True)
    if args.smoke:
        tx, ty, vx, vy = tx[:256], ty[:256], vx[:512], vy[:512]; args.epochs = 2
    model = QVFScratch(C, sham=args.sham).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={nparam}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        m = evaluate(model, vx, vy, device, C); best = max(best, m["auc"])
        ai = cn.index("axion") if "axion" in cn else 0
        axr = m["cm"][ai, ai] / m["cm"][ai].sum()
        print(f"[ep {ep+1:02d}] val AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} axion={axr:.4f}",
              flush=True)
    print(f"\n[DONE] FROM-SCRATCH QVF {mode} best AUC={best:.4f} params={nparam}")


if __name__ == "__main__":
    main()
