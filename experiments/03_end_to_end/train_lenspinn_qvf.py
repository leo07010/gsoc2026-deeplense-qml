#!/usr/bin/env python
"""
EXPERIMENT A — LensPINN-physics features + QVF quantum HEAD.

Combines LensPINN's physics-informed preprocessing (Ojha/Gleyzer/Toomey/Reddy,
NeurIPS ML4PS 2024) with the QVF quantum readout that we verified beats its sham
in the low-data regime. Tested at LensPINN's native data scale (≈2400/class),
where the quantum advantage is visible (not suppressed by the 0.99 ceiling).

  image ─┬─ raw ───────────────────────────┐
         └─ physics-prep tanh(∇²[log(Imax/I)]²) ┤ 2-ch → CNN → feature(128)
                                              → head ∈ {quantum, sham, classical}
  quantum : NAE(energy→256 Boltzmann amps) → 8-qubit amplitude-embed + PQC → ⟨Z⟩
  sham    : NAE → classical Linear(256→K)+tanh        (matched, no circuit)
  classical: plain Linear(128→3)                       (no NAE, no circuit)
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from quantum_qvf import _circuit, enc_shape, K_LATENT, DIM        # 8-qubit QVF circuit


def physics_prep(x):
    """LensPINN-style: contrast (log) → square → Laplacian (2nd-deriv edges) → tanh."""
    eps = 1e-3
    imax = x.amax(dim=(2, 3), keepdim=True)
    L = torch.log((imax + eps) / (x + eps)) ** 2
    lap = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device).view(1, 1, 3, 3)
    return torch.tanh(F.conv2d(L, lap, padding=1))


class CNNEncoder(nn.Module):
    def __init__(self, in_ch=2, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class NAE(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.energy = nn.Sequential(nn.Linear(in_dim, hid), nn.Tanh(), nn.Linear(hid, DIM))

    def forward(self, x):
        return torch.sqrt(torch.softmax(-self.energy(x), dim=1) + 1e-12)


class LensQVF(nn.Module):
    def __init__(self, head, n_classes=3, feat=128):
        super().__init__()
        self.head_kind = head
        self.cnn = CNNEncoder(2, feat)
        if head == "classical":
            self.out = nn.Linear(feat, n_classes)
        else:
            self.nae = NAE(feat)
            if head == "quantum":
                self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
            else:                                            # sham
                self.cl = nn.Linear(DIM, K_LATENT)
            self.out = nn.Sequential(nn.LayerNorm(K_LATENT), nn.Linear(K_LATENT, n_classes))

    def quantum_params(self):
        return [self.w] if self.head_kind == "quantum" else []

    def forward(self, x):
        feat = self.cnn(torch.cat([x, physics_prep(x)], dim=1))
        if self.head_kind == "classical":
            return self.out(feat)
        amp = self.nae(feat)
        z = torch.stack(_circuit(amp, self.w), -1).to(amp.dtype) if self.head_kind == "quantum" \
            else torch.tanh(self.cl(amp))
        return self.out(z)


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def subsample(ty, N, C, seed):
    r = np.random.default_rng(500 + seed)
    return np.concatenate([r.choice(np.where(ty == c)[0], min(N, (ty == c).sum()),
                                    replace=False) for c in range(C)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", choices=["quantum", "sham", "classical"], required=True)
    ap.add_argument("--n_per_class", type=int, default=2400)      # LensPINN regime
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qlr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    TX = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    TY = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    if args.n_per_class > 0:
        idx = subsample(TY.numpy(), args.n_per_class, C, args.seed); TX, TY = TX[idx], TY[idx]
    if args.smoke:
        TX, TY, vx, vy = TX[:400], TY[:400], vx[:800], vy[:800]; args.epochs = 3
    print(f"[INFO] LensPINN-QVF head={args.head} | {os.path.basename(args.data)} "
          f"N={args.n_per_class} train{tuple(TX.shape)}", flush=True)
    model = LensQVF(args.head, C).to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": args.lr, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": args.qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] params={nt}", flush=True)
    loader = DataLoader(TensorDataset(TX, TY), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        a = evaluate(model, vx, vy, device, C); best = max(best, a)
        print(f"[ep {ep+1:02d}] AUC={a:.4f}", flush=True)
    print(f"\n[DONE] LensPINN-QVF head={args.head} N={args.n_per_class} "
          f"best AUC={best:.4f} params={nt}")


if __name__ == "__main__":
    main()
