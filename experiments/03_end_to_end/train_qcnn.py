#!/usr/bin/env python
"""
EXPERIMENT B — QCNN: quantum replaces the FEATURE EXTRACTOR (QVF-style
quanvolution), vs a parameter-matched classical convolution.

This is the placement our analysis predicts will TIE/LOSE (surrogate theorem;
QViT confirmed quantum-in-encoder loses) — we test it cleanly anyway.

  image 64×64 → area-pool to 32×32 → unfold 4×4 patches (stride 4) → 64 patches × 16 px
   quantum : per patch  amplitude-embed 16 px into 4 qubits → trainable PQC
             → 4 ⟨Z⟩  → feature map (B,4,8,8)
   sham    : per patch  Linear(16→4)+tanh  (matched dim, no circuit)
  → small classical CNN head → 3 classes.   Both share the SAME head.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

NQ = 4                                       # 2^4 = 16 = 4x4 patch
NL = 3
_DEV = qml.device("default.qubit", wires=NQ)
_SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=NL, n_wires=NQ)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _quanv(amp, weights):                    # amp (M,16) → 4 ⟨Z⟩
    qml.AmplitudeEmbedding(amp, wires=range(NQ), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(NQ))
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]


def patches(x):                              # x (B,1,64,64) → (B,64,16) 4x4 patches of 32x32
    x = F.adaptive_avg_pool2d(x, 32)
    p = F.unfold(x, kernel_size=4, stride=4)          # (B,16,64)
    return p.transpose(1, 2)                           # (B,64,16)


class QCNN(nn.Module):
    def __init__(self, quantum, n_classes=3):
        super().__init__()
        self.quantum = quantum
        if quantum:
            self.w = nn.Parameter(0.1 * torch.randn(_SHAPE))
        else:
            self.cl = nn.Linear(16, NQ)               # matched classical patch map
        self.head = nn.Sequential(                     # shared classical head
            nn.Conv2d(NQ, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.out = nn.Linear(64, n_classes)

    def quantum_params(self):
        return [self.w] if self.quantum else []

    def forward(self, x):
        B = x.shape[0]
        p = patches(x)                                 # (B,64,16)
        flat = p.reshape(B * 64, 16)
        if self.quantum:
            z = torch.stack(_quanv(flat, self.w), -1).to(x.dtype)   # (B*64,4)
        else:
            z = torch.tanh(self.cl(flat))
        fmap = z.reshape(B, 8, 8, NQ).permute(0, 3, 1, 2)           # (B,4,8,8)
        return self.out(self.head(fmap).flatten(1))


def evaluate(model, x, y, device, C, bs=256):
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
    ap.add_argument("--arm", choices=["quantum", "classical"], required=True)
    ap.add_argument("--n_per_class", type=int, default=2400)
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
    print(f"[INFO] QCNN arm={args.arm} | {os.path.basename(args.data)} "
          f"N={args.n_per_class} train{tuple(TX.shape)}", flush=True)
    model = QCNN(args.arm == "quantum", C).to(device)
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
    print(f"\n[DONE] QCNN arm={args.arm} N={args.n_per_class} best AUC={best:.4f} params={nt}")


if __name__ == "__main__":
    main()
