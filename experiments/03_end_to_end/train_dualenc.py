#!/usr/bin/env python
"""
Dual-encoder hybrid: a classical CNN encoder and a quantum encoder run in
PARALLEL on the same image; their latents are merged by a learnable FUSION HEAD
(not a bare concat) and classified. Everything trains end-to-end from scratch.

  image ─┬─→ CNN encoder ─────────────→ z_c (d_c)
         └─→ quantum encoder (QCT|QVF) → z_q (d_q)
                                          │
               fusion head {concat|gated|film} → logits

Quantum encoder:
  --qmode qct : global feature → 8 angles → RY → StronglyEntangling → 8⟨Z⟩+8⟨ZZ⟩
  --qmode qvf : global feature → neural amplitude encoding (learnable energy →
                256 Boltzmann amplitudes) → StronglyEntangling → 8⟨Z⟩

Fusion head (--fuse):
  concat : [LN(P_c z_c) ; LN(P_q z_q)] → MLP → logits
  gated  : h = P_c z_c + tanh(gate) ⊙ P_q z_q → head      (gate learnable)
  film   : (γ,β) = g(z_q); h = γ ⊙ P_c z_c + β → head     (quantum MODULATES
           the classical features — cannot be trivially ignored)

Controls:
  --sham      : quantum encoder → classical Linear(feat→d_q)+tanh (matched dim,
                NO circuit) — isolates the circuit's contribution.
  zero-q ablation: every eval also reports AUC with z_q forced to 0, proving
                whether the quantum latent is actually used by the fusion head.

Data: model_X.npz from cache_model.py.  Backend: default.qubit + backprop.
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

N_Q = 8
N_LAYERS = 4
DIM = 2 ** N_Q                                   # 256
_DEV = qml.device("default.qubit", wires=N_Q)
_SEL_SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _qct_circuit(angles, weights):               # angles (B,8) → 16 expvals
    for i in range(N_Q):
        qml.RY(angles[..., i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)] + \
           [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _qvf_circuit(amp, weights):                  # amp (B,256) → 8 expvals
    qml.AmplitudeEmbedding(amp, wires=range(N_Q), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)]


class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)            # (B, out_dim)


class QuantumEncoder(nn.Module):
    """Produces z_q (d_q) from a classical feature vector via a PQC (or sham)."""

    def __init__(self, feat_dim, qmode, sham, e_hid=128):
        super().__init__()
        self.qmode = qmode
        self.sham = sham
        self.d_q = 2 * N_Q if qmode == "qct" else N_Q
        if qmode == "qct":
            self.proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, N_Q))
        else:                                    # qvf: neural amplitude encoding
            self.energy = nn.Sequential(nn.LayerNorm(feat_dim),
                                        nn.Linear(feat_dim, e_hid), nn.Tanh(),
                                        nn.Linear(e_hid, DIM))
        if sham:
            in_d = N_Q if qmode == "qct" else DIM
            self.cl = nn.Linear(in_d, self.d_q)
        else:
            self.w = nn.Parameter(0.01 * torch.randn(_SEL_SHAPE))

    def forward(self, feat):
        if self.qmode == "qct":
            pre = torch.tanh(self.proj(feat)) * np.pi          # angles
            if self.sham:
                return torch.tanh(self.cl(pre))
            return torch.stack(_qct_circuit(pre, self.w), -1).to(feat.dtype)
        else:
            amp = torch.sqrt(torch.softmax(-self.energy(feat), dim=1) + 1e-12)
            if self.sham:
                return torch.tanh(self.cl(amp))
            return torch.stack(_qvf_circuit(amp, self.w), -1).to(feat.dtype)


class FusionHead(nn.Module):
    def __init__(self, d_c, d_q, fuse, d=64, n_classes=3):
        super().__init__()
        self.fuse = fuse
        self.pc = nn.Linear(d_c, d)
        if fuse == "concat":
            self.pq = nn.Linear(d_q, d)
            self.lnc, self.lnq = nn.LayerNorm(d), nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_classes))
        elif fuse == "gated":
            self.pq = nn.Linear(d_q, d)
            self.gate = nn.Parameter(torch.zeros(d))           # starts at classical-only
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        else:                                                  # film
            self.film = nn.Linear(d_q, 2 * d)                  # → (γ, β)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, z_c, z_q):
        hc = self.pc(z_c)
        if self.fuse == "concat":
            h = torch.cat([self.lnc(hc), self.lnq(self.pq(z_q))], -1)
        elif self.fuse == "gated":
            h = hc + torch.tanh(self.gate) * self.pq(z_q)
        else:                                                  # film
            g, b = self.film(z_q).chunk(2, -1)
            h = (1 + g) * hc + b
        return self.head(h)


class DualEncoder(nn.Module):
    def __init__(self, qmode, fuse, sham, feat=128, n_classes=3):
        super().__init__()
        self.cnn = CNNEncoder(feat)
        self.qenc = QuantumEncoder(feat, qmode, sham)
        self.fusion = FusionHead(feat, self.qenc.d_q, fuse, n_classes=n_classes)

    def forward(self, x, zero_q=False):
        feat = self.cnn(x)
        z_q = self.qenc(feat)
        if zero_q:
            z_q = torch.zeros_like(z_q)
        return self.fusion(feat, z_q)


def evaluate(model, x, y, device, C, zero_q=False, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device), zero_q=zero_q).cpu())
    probs = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    return dict(auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
                acc=accuracy_score(yy, probs.argmax(1)),
                f1=f1_score(yy, probs.argmax(1), average="macro"),
                cm=confusion_matrix(yy, probs.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--qmode", choices=["qct", "qvf"], default="qct")
    ap.add_argument("--fuse", choices=["concat", "gated", "film"], default="film")
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    mode = "SHAM" if args.sham else "QUANTUM"
    if args.smoke:
        rng = np.random.default_rng(0)           # data is class-ordered → sample randomly
        ti = rng.choice(len(tx), 1500, replace=False); vi = rng.choice(len(vx), 1500, replace=False)
        tx, ty, vx, vy = tx[ti], ty[ti], vx[vi], vy[vi]; args.epochs = 3
    print(f"[INFO] DUAL-ENC {mode} qmode={args.qmode} fuse={args.fuse} | "
          f"data={os.path.basename(args.data)} train{tuple(tx.shape)} classes={cn}", flush=True)
    model = DualEncoder(args.qmode, args.fuse, args.sham, n_classes=C).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={nparam} d_q={model.qenc.d_q}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
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
        print(f"[ep {ep+1:02d}] AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} axion={axr:.4f}",
              flush=True)
    mz = evaluate(model, vx, vy, device, C, zero_q=True)        # quantum-OFF ablation
    print(f"\n[DONE] DUAL-ENC {mode} qmode={args.qmode} fuse={args.fuse} "
          f"best AUC={best:.4f} | zero-q AUC={mz['auc']:.4f} "
          f"(Δ used-vs-off = {best - mz['auc']:+.4f}) params={nparam}")


if __name__ == "__main__":
    main()
