#!/usr/bin/env python
"""
QUANTUM MULTI-VIEW FUSION for strong-lensing substructure classification.

Synthesis of the literature: quantum-classical hybrids tie on SINGLE-modality
discriminative tasks (Bowles 2024; Schnabel 2025; Freinberger 2026; our own
nulls), but a capacity-matched quantum FUSION layer wins in the HIGH-MODALITY
regime, with the advantage GROWING in the number of modalities (QFL,
arXiv:2510.06938). Lensing is natively single-image, so we synthesize M
physically-motivated "views" (multi-scale / multi-derivative maps) and fuse
them with a QFL-style quantum fusion layer — the regime where quantum fusion
is documented to help.

Central hypothesis (pre-registered): the quantum-minus-sham AUC gap is an
increasing function of the view count M. At M=1 they tie (reproducing every
prior null); at large M quantum > sham AND > classical concat fusion.

Views (each → shared-arch small CNN → d-dim latent), all GPU on-the-fly:
  raw, sobel-grad, laplacian, fft-log-mag, highpass-σ1, highpass-σ2,
  local-std, lowpass        (use first M)

Fusion (--fuse), all sharing the SAME per-view encoders and the SAME head:
  quantum : concat latents → proj → n_q angles → P× [re-upload + entangle]
            → measure ⟨Z⟩,⟨ZZ⟩  (QFL: high-order cross-view interaction,
            LINEAR params; pathology-fixed circuit: wd=0, U(0,π) init, own lr)
  sham    : same proj → classical MLP (>= circuit params) → N_OBS  (the
            capacity-matched control QFL itself never ran)
  concat  : concat latents → MLP → N_OBS            (standard fusion baseline)
  lowrank : low-rank tensor fusion (LMF-style), rank R                 (classical)
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
import pennylane as qml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

N_Q = 8
N_OBS = 2 * N_Q
_DEV = qml.device("default.qubit", wires=N_Q)


# ───────────────── physically-motivated views (no grad needed) ─────────────────
def _gauss(sigma, ks, device):
    ax = torch.arange(ks, device=device) - ks // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).view(1, 1, ks, ks)


def _conv(x, k):
    return F.conv2d(x, k, padding=k.shape[-1] // 2)


def make_views(x, M):
    """x (B,1,64,64) → list of M view tensors (B,1,64,64), normalized per-view."""
    dev = x.device
    sob = torch.tensor([[1., 0, -1], [2, 0, -2], [1, 0, -1]], device=dev).view(1, 1, 3, 3)
    lap = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=dev).view(1, 1, 3, 3)
    g1, g2 = _gauss(1.0, 5, dev), _gauss(2.0, 9, dev)
    gx, gy = _conv(x, sob), _conv(x, sob.transpose(-1, -2))
    fft = torch.log1p(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs())
    mean1 = _conv(x, g1)
    var = (_conv(x ** 2, g1) - mean1 ** 2).clamp(min=0).sqrt()
    views = [
        x,                                          # raw arc
        (gx ** 2 + gy ** 2).sqrt(),                 # gradient magnitude (edges)
        _conv(x, lap).abs(),                        # laplacian (fine substructure)
        fft,                                        # power spectrum
        x - mean1,                                  # high-pass σ1 (subhalo scale)
        x - _conv(x, g2),                           # high-pass σ2 (larger scale)
        var,                                        # local std (texture)
        mean1,                                      # low-pass (smooth lens)
    ][:M]
    out = []
    for v in views:                                 # per-view min-max to [0,1]
        vmin = v.amin(dim=(2, 3), keepdim=True)
        vmax = v.amax(dim=(2, 3), keepdim=True)
        out.append((v - vmin) / (vmax - vmin + 1e-6))
    return out


# ───────────────── per-view encoder ─────────────────
class ViewEncoder(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, d, 3, 2, 1), nn.BatchNorm2d(d), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)               # (B,d)


# ───────────────── QFL-style quantum fusion circuit ─────────────────
@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _fuse_circuit(angles, weights, P):
    for p in range(P):
        for i in range(N_Q):                        # data re-upload (degree ↑)
            qml.RY(angles[..., i], wires=i)
        for i in range(N_Q):
            qml.RZ(weights[p, i, 0], wires=i)
            qml.RY(weights[p, i, 1], wires=i)
        for i in range(N_Q):
            qml.CNOT(wires=[i, (i + 1) % N_Q])
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)] + \
           [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]


class QFusion(nn.Module):
    def __init__(self, in_dim, P):
        super().__init__()
        self.P = P
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        self.w = nn.Parameter(np.pi * torch.rand(P, N_Q, 2))     # U(0,π) init
    def quantum_params(self): return [self.w]
    def forward(self, cat):
        ang = torch.tanh(self.proj(cat)) * np.pi
        return torch.stack(_fuse_circuit(ang, self.w, self.P), -1).to(cat.dtype)


class ShamFusion(nn.Module):
    """Capacity-matched classical core: same proj, MLP with >= circuit params."""
    def __init__(self, in_dim, P):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        h = max(N_OBS, P * N_Q)                       # >= circuit param budget
        self.core = nn.Sequential(nn.Linear(N_Q, h), nn.Tanh(), nn.Linear(h, N_OBS), nn.Tanh())
    def quantum_params(self): return []
    def forward(self, cat):
        return self.core(torch.tanh(self.proj(cat)) * np.pi)


class ConcatFusion(nn.Module):
    def __init__(self, in_dim, P):
        super().__init__()
        self.core = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64),
                                  nn.ReLU(), nn.Linear(64, N_OBS))
    def quantum_params(self): return []
    def forward(self, cat): return self.core(cat)


class LowRankFusion(nn.Module):
    """LMF-style low-rank multimodal fusion over M views (rank R)."""
    def __init__(self, M, d, R=4):
        super().__init__()
        self.M, self.d, self.R = M, d, R
        self.factors = nn.Parameter(0.1 * torch.randn(M, R, d + 1, N_OBS))
    def quantum_params(self): return []
    def forward_lats(self, lats):
        out = torch.ones(lats[0].shape[0], self.R, N_OBS, device=lats[0].device)
        for m, z in enumerate(lats):
            z1 = torch.cat([z, torch.ones(z.shape[0], 1, device=z.device)], 1)  # (B,d+1)
            out = out * torch.einsum("bi,rio->bro", z1, self.factors[m])
        return out.sum(1)                              # (B,N_OBS)


class QFusionNet(nn.Module):
    def __init__(self, M, fuse, P, d=16, n_classes=3):
        super().__init__()
        self.M, self.fuse_kind = M, fuse
        self.encs = nn.ModuleList([ViewEncoder(d) for _ in range(M)])
        if fuse == "lowrank":
            self.fuse = LowRankFusion(M, d)
        else:
            cls = {"quantum": QFusion, "sham": ShamFusion, "concat": ConcatFusion}[fuse]
            self.fuse = cls(M * d, P)
        self.head = nn.Sequential(nn.LayerNorm(N_OBS), nn.Linear(N_OBS, n_classes))

    def quantum_params(self):
        return self.fuse.quantum_params() if hasattr(self.fuse, "quantum_params") else []

    def forward(self, x):
        with torch.no_grad():
            views = make_views(x, self.M)
        lats = [enc(v) for enc, v in zip(self.encs, views)]
        if self.fuse_kind == "lowrank":
            z = self.fuse.forward_lats(lats)
        else:
            z = self.fuse(torch.cat(lats, 1))
        return self.head(z)


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fuse", choices=["quantum", "sham", "concat", "lowrank"], required=True)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
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
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    if args.smoke:
        rng = np.random.default_rng(0)
        ti = rng.choice(len(tx), 1500, replace=False); vi = rng.choice(len(vx), 1500, replace=False)
        tx, ty, vx, vy = tx[ti], ty[ti], vx[vi], vy[vi]; args.epochs = 3
    print(f"[INFO] QFUSION fuse={args.fuse} M={args.M} P={args.P} | "
          f"data={os.path.basename(args.data)} train{tuple(tx.shape)}", flush=True)
    model = QFusionNet(args.M, args.fuse, args.P, n_classes=C).to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": args.lr, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": args.qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    nf = sum(p.numel() for p in model.fuse.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] fusion params={nf} total params={nt}", flush=True)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        a = evaluate(model, vx, vy, device, C); best = max(best, a)
        print(f"[ep {ep+1:02d}] AUC={a:.4f}", flush=True)
    print(f"\n[DONE] QFUSION fuse={args.fuse} M={args.M} P={args.P} "
          f"best AUC={best:.4f} fusion_params={nf}")


if __name__ == "__main__":
    main()
