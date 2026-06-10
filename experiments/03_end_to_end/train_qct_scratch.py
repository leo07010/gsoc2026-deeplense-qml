#!/usr/bin/env python
"""
From-SCRATCH Quantum-Classical Transformer (QCT) on raw 64x64 lensing images.

Unlike all earlier experiments (which bolted quantum onto FROZEN MAE features),
here the classical CNN encoder AND the quantum head are trained TOGETHER from
scratch — so the quantum branch can help shape the representation from the start.

  image 64x64 → Conv patch-embed → 64 classical patch tokens ─┐
  global-pooled feature → angles → PQC → 2Q quantum tokens ────┤→ mixed
  + learnable CLS + type embeddings                            │  self-attn
  [CLS ; patch tokens ; quantum tokens] → Transformer → CLS → 3-class
  --sham : quantum tokens → classical projection (matched dim)

Data: model_X.npz from cache_model.py (train_x/val_x 64x64, labels, class_names).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
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

N_Q = 8
N_LAYERS = 4
N_WEIGHTS = N_Q * 2 * N_LAYERS
N_TOK = 2 * N_Q                 # 8 ⟨Z⟩ + 8 ring ⟨ZZ⟩ = 16
D = 64
N_HEADS = 4
N_TBLOCKS = 2
_DEV = qml.device("default.qubit", wires=N_Q)
_REUPLOAD = os.environ.get("QF_REUPLOAD", "0") == "1"
_SHAM = os.environ.get("QF_SHAM", "0") == "1"


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(angles, weights):
    if not _REUPLOAD:
        for i in range(N_Q):
            qml.RY(angles[..., i], wires=i)
    for L in range(N_LAYERS):
        if _REUPLOAD:
            for i in range(N_Q):
                qml.RY(angles[..., i], wires=i)
        b = L * N_Q * 2
        for i in range(N_Q):
            qml.RZ(weights[b + i * 2], wires=i); qml.RY(weights[b + i * 2 + 1], wires=i)
        for i in range(N_Q - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_Q - 1, 0])
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)] + \
           [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]


class QCTScratch(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.stem = nn.Conv2d(1, D, kernel_size=8, stride=8)        # 64x64 → 8x8=64 tokens
        self.qproj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, N_Q))
        if _SHAM:
            self.sham = nn.Linear(D, N_TOK)
        else:
            self.qweights = nn.Parameter(0.01 * torch.randn(N_WEIGHTS))
        self.tok_embed = nn.Linear(1, D)
        self.tok_id = nn.Parameter(0.02 * torch.randn(N_TOK, D))
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, D))
        self.type_emb = nn.Parameter(0.02 * torch.randn(3, D))      # cls/patch/quantum
        blk = nn.TransformerEncoderLayer(D, N_HEADS, 4 * D, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(blk, N_TBLOCKS)
        self.norm = nn.LayerNorm(D); self.out = nn.Linear(D, n_classes)

    def forward(self, x):                                            # x (B,1,64,64)
        B = x.shape[0]
        p = self.stem(x).flatten(2).transpose(1, 2)                  # (B,64,D)
        p = p + self.type_emb[1]
        g = p.mean(1)                                                # (B,D) global
        if _SHAM:
            qz = torch.tanh(self.sham(g))
        else:
            ang = torch.tanh(self.qproj(g)) * np.pi
            qz = torch.stack(_circuit(ang, self.qweights), dim=-1).to(x.dtype)
        qtok = self.tok_embed(qz.unsqueeze(-1)) + self.tok_id + self.type_emb[2]
        cls = self.cls.expand(B, -1, -1) + self.type_emb[0]
        seq = torch.cat([cls, p, qtok], dim=1)
        return self.out(self.norm(self.enc(seq)[:, 0]))


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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)        # (N,1,64,64)
    ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    mode = "SHAM" if _SHAM else "QUANTUM"
    print(f"[INFO] FROM-SCRATCH QCT {mode} reupload={_REUPLOAD} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)} classes={cn}", flush=True)
    if args.smoke:
        tx, ty, vx, vy = tx[:256], ty[:256], vx[:512], vy[:512]; args.epochs = 2
    model = QCTScratch(C).to(device)
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
    print(f"\n[DONE] {mode} best AUC={best:.4f} params={nparam}")


if __name__ == "__main__":
    main()
