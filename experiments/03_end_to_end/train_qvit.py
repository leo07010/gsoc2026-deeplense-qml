#!/usr/bin/env python
"""
QUANTUM INSIDE THE ENCODER (not as the readout head). A compact ViT with a
quantum mixing layer injected at mid-depth, so the quantum component PARTICIPATES
IN REPRESENTATION LEARNING — subsequent transformer blocks refine its output, and
gradients flow through it to shape what earlier layers learn. This is the one
placement untested under our sham methodology.

  patch-embed → +CLS → [blocks 0..qpos-1] → QUANTUM MIX on CLS (residual)
              → [blocks qpos..L-1] → LN(CLS) → Linear → 3 classes

--fuse:
  quantum : CLS → angles → PQC (re-upload×P) → ⟨Z⟩,⟨ZZ⟩ → proj-back → +CLS
  sham    : CLS → angles → classical MLP (matched params) → proj-back → +CLS
  none    : pure classical ViT (no injected layer)         — reference ceiling

Trained end-to-end from scratch. Capacity-matched sham isolates the circuit.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import argparse
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

N_Q, P = 8, 2
N_OBS = 2 * N_Q
_DEV = qml.device("default.qubit", wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(angles, weights):
    for p in range(P):
        for i in range(N_Q):
            qml.RY(angles[..., i], wires=i)
        for i in range(N_Q):
            qml.RZ(weights[p, i, 0], wires=i); qml.RY(weights[p, i, 1], wires=i)
        for i in range(N_Q):
            qml.CNOT(wires=[i, (i + 1) % N_Q])
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)] + \
           [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]


class QuantumMix(nn.Module):
    """Mid-encoder layer: refines the CLS token via a circuit, residual-added."""
    def __init__(self, D, fuse):
        super().__init__()
        self.fuse = fuse
        self.to_ang = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, N_Q))
        if fuse == "quantum":
            self.w = nn.Parameter(np.pi * torch.rand(P, N_Q, 2))
        else:                                          # sham: matched-param MLP
            h = max(N_OBS, P * N_Q)
            self.core = nn.Sequential(nn.Linear(N_Q, h), nn.Tanh(), nn.Linear(h, N_OBS), nn.Tanh())
        self.back = nn.Linear(N_OBS, D)
        nn.init.zeros_(self.back.weight); nn.init.zeros_(self.back.bias)  # start ≈ identity

    def quantum_params(self):
        return [self.w] if self.fuse == "quantum" else []

    def forward(self, cls):
        ang = torch.tanh(self.to_ang(cls)) * np.pi
        if self.fuse == "quantum":
            z = torch.stack(_circuit(ang, self.w), -1).to(cls.dtype)
        else:
            z = self.core(ang)
        return cls + self.back(z)                      # residual injection


class QViT(nn.Module):
    def __init__(self, fuse, n_classes=3, D=64, depth=4, heads=4, qpos=2):
        super().__init__()
        self.fuse = fuse
        self.stem = nn.Conv2d(1, D, kernel_size=8, stride=8)     # 64→8x8=64 tokens
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, D))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 65, D))
        blk = lambda: nn.TransformerEncoderLayer(D, heads, 4 * D, dropout=0.1,
                                                 batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([blk() for _ in range(depth)])
        self.qpos = qpos
        self.qmix = QuantumMix(D, fuse) if fuse != "none" else None
        self.norm = nn.LayerNorm(D); self.out = nn.Linear(D, n_classes)

    def quantum_params(self):
        return self.qmix.quantum_params() if self.qmix else []

    def forward(self, x):
        B = x.shape[0]
        p = self.stem(x).flatten(2).transpose(1, 2)             # (B,64,D)
        seq = torch.cat([self.cls.expand(B, -1, -1), p], 1) + self.pos
        for i, b in enumerate(self.blocks):
            seq = b(seq)
            if self.qmix is not None and i == self.qpos - 1:    # inject mid-stack
                cls = self.qmix(seq[:, 0])
                seq = torch.cat([cls.unsqueeze(1), seq[:, 1:]], 1)
        return self.out(self.norm(seq[:, 0]))


def evaluate(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, p, average="macro", multi_class="ovr"),
                acc=accuracy_score(yy, p.argmax(1)),
                f1=f1_score(yy, p.argmax(1), average="macro"),
                cm=confusion_matrix(yy, p.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fuse", choices=["quantum", "sham", "none"], required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
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
    print(f"[INFO] QViT fuse={args.fuse} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)}", flush=True)
    model = QViT(args.fuse, C).to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": args.lr, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": args.qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] total params={nt} qpos={model.qpos}", flush=True)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    ai = cn.index("axion") if "axion" in cn else 0
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        m = evaluate(model, vx, vy, device, C); best = max(best, m["auc"])
        axr = m["cm"][ai, ai] / max(m["cm"][ai].sum(), 1)
        print(f"[ep {ep+1:02d}] AUC={m['auc']:.4f} acc={m['acc']:.4f} axion={axr:.4f}", flush=True)
    print(f"\n[DONE] QViT fuse={args.fuse} on {os.path.basename(args.data)} "
          f"best AUC={best:.4f} params={nt}")


if __name__ == "__main__":
    main()
