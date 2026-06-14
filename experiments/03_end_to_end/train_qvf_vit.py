#!/usr/bin/env python
"""
Quantum-enhanced ViT: the verified QVF quantum HEAD on a ViT encoder (instead of
the small CNN of QVF-scratch). Tests whether the QVF quantum advantage measured
on CNN transfers to stronger ViT features, in the low-data regime where it shows.

  image → ViT encoder (mainv2, 6 blocks, 192-d) → CLS(192)
        → NAE(energy→256 Boltzmann amps)
        → head ∈ {quantum: amplitude-embed+PQC→⟨Z⟩ ; sham: Linear(256→8)+tanh}
        → LN → Linear → 3 classes
Same NAE / circuit / head / recipe (AdamW, qlr=1e-2) as QVF-scratch — only the
encoder is ViT not CNN. From-scratch (no MAE pretrain) for a clean encoder swap.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "jax" not in sys.modules:
    sys.modules["jax"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from mainv2 import ViTEncoder
from quantum_qvf import _circuit, enc_shape, K_LATENT, DIM        # 8-qubit QVF head


class NAE(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.energy = nn.Sequential(nn.Linear(in_dim, hid), nn.Tanh(), nn.Linear(hid, DIM))

    def forward(self, x):
        return torch.sqrt(torch.softmax(-self.energy(x), dim=1) + 1e-12)


class QVFViT(nn.Module):
    def __init__(self, head, n_classes=3):
        super().__init__()
        self.head_kind = head
        self.enc = ViTEncoder(img_size=64, patch_size=4, in_chans=1, embed_dim=192,
                              depth=6, num_heads=3, mlp_ratio=4.0, drop_rate=0.1)
        self.nae = NAE(192)
        if head == "quantum":
            self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
        else:
            self.cl = nn.Linear(DIM, K_LATENT)
        self.out = nn.Sequential(nn.LayerNorm(K_LATENT), nn.Linear(K_LATENT, n_classes))

    def load_encoder(self, path):
        self.enc.load_state_dict(torch.load(path, map_location="cpu"))

    def quantum_params(self):
        return [self.w] if self.head_kind == "quantum" else []

    def forward(self, x):
        cls = self.enc(x)[:, 0]                       # ViT CLS token (B,192)
        amp = self.nae(cls)
        z = torch.stack(_circuit(amp, self.w), -1).to(amp.dtype) if self.head_kind == "quantum" \
            else torch.tanh(self.cl(amp))
        return self.out(z)


def auc_of(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy()
    yb = label_binarize(y.numpy(), classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def subsample(ty, N, C, seed):
    r = np.random.default_rng(500 + seed)
    return np.concatenate([r.choice(np.where(ty == c)[0], min(N, (ty == c).sum()),
                                    replace=False) for c in range(C)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", choices=["quantum", "sham"], required=True)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc_lr", type=float, default=5e-5)
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
        ti = np.random.default_rng(0).choice(len(TX), min(600, len(TX)), replace=False)
        vi = np.random.default_rng(1).choice(len(vx), 1200, replace=False)
        TX, TY, vx, vy = TX[ti], TY[ti], vx[vi], vy[vi]; args.epochs = 3
    print(f"[INFO] QVF-ViT head={args.head} | {os.path.basename(args.data)} "
          f"N={args.n_per_class} train{tuple(TX.shape)}", flush=True)
    model = QVFViT(args.head, C)
    if args.pretrained:
        model.load_encoder(args.pretrained); print(f"[INFO] loaded encoder {args.pretrained}", flush=True)
    model = model.to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    enc_p = [p for p in model.enc.parameters()]; encids = {id(p) for p in enc_p}
    head_p = [p for p in model.parameters() if id(p) not in qids and id(p) not in encids]
    elr = args.enc_lr if args.pretrained else args.lr      # pretrained encoder: gentle lr
    groups = [{"params": enc_p, "lr": elr, "weight_decay": 1e-4},
              {"params": head_p, "lr": args.lr, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": args.qlr, "weight_decay": 0.0})
    print(f"[INFO] lr: encoder={elr} head={args.lr} circuit={args.qlr if qp else '-'}", flush=True)
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
        a = auc_of(model, vx, vy, device, C); best = max(best, a)
        print(f"[ep {ep+1:02d}] AUC={a:.4f}", flush=True)
    print(f"\n[DONE] QVF-ViT head={args.head} N={args.n_per_class} best AUC={best:.4f} params={nt}")


if __name__ == "__main__":
    main()
