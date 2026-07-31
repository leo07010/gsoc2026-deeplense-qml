#!/usr/bin/env python
"""QVF-HYBRID-NOENT: the hybrid architecture's entanglement ablation arm.

Identical to train_qvf_opt.py --nq 8 (h(128) ⊕ m(52) residual hybrid,
143,655 params, same NAE / wide readout / LN / head / protocol) EXCEPT the
circuit: StronglyEntanglingLayers is replaced by the SAME 4x8x3=96 rotation
angles applied as per-qubit qml.Rot with NO CNOT ring (mirrors
train_qvf_noent.py). Completes the three-arm mechanism test INSIDE the final
architecture: hyb-ENT vs hyb-NOENT vs hyb-sham at the claim cells.

Interpretation (pre-registered):
  hyb-NOENT ~= hyb-sham   -> entanglement carries the residual low-N gain.
  hyb-NOENT ~= hyb-ENT    -> the gain is probability-marginal structure only.

Protocol identical to scratch/wide/hybrid/opt: AdamW 1e-3 wd=1e-4, cosine,
ls=0.05, batch 128, subsample rng 1000+seed, best-val-AUC, 20 ep.
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
import pennylane as qml

N_Q      = 8
DIM      = 2 ** N_Q
N_LAYERS = 4                                                          # 96 angles
PAIRS    = [(i, j) for i in range(N_Q) for j in range(i + 1, N_Q)]    # 28
K_WIDE   = 3 * N_Q + len(PAIRS)                                       # 52

_DEV = qml.device("default.qubit", wires=N_Q)


def enc_shape():
    return (N_LAYERS, N_Q, 3)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit_noent_wide(amp, weights):
    qml.AmplitudeEmbedding(amp, wires=range(N_Q), normalize=True)
    for l in range(N_LAYERS):                       # same 96 angles, NO CNOTs
        for q in range(N_Q):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
    obs  = [qml.expval(qml.PauliZ(q)) for q in range(N_Q)]
    obs += [qml.expval(qml.PauliX(q)) for q in range(N_Q)]
    obs += [qml.expval(qml.PauliY(q)) for q in range(N_Q)]
    obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in PAIRS]
    return obs


class CNNEncoder(nn.Module):
    """Identical to train_qvf_scratch.CNNEncoder / qvf_plain_baseline."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
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


class QVFHybNoent(nn.Module):
    def __init__(self, n_classes=3, feat=128):
        super().__init__()
        self.cnn = CNNEncoder(feat)
        self.nae = NAE(feat)
        self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
        self.ln_h = nn.LayerNorm(feat)
        self.ln_m = nn.LayerNorm(K_WIDE)
        self.head = nn.Linear(feat + K_WIDE, n_classes)

    def forward(self, x):
        h = self.cnn(x)
        amp = self.nae(h)
        m = torch.stack(_circuit_noent_wide(amp, self.w), dim=-1).to(amp.dtype)
        return self.head(torch.cat([self.ln_h(h), self.ln_m(m)], dim=-1))


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
    print(f"[INFO] QVF-HYB-NOENT K={K_WIDE} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)}", flush=True)
    if args.n_per_class > 0:
        rng = np.random.default_rng(1000 + args.seed); idx = []
        for c in range(C):
            ci = np.where(ty.numpy() == c)[0]
            idx.extend(rng.choice(ci, min(args.n_per_class, len(ci)), replace=False))
        rng.shuffle(idx); tx, ty = tx[idx], ty[idx]
        print(f"[INFO] subsampled to N={args.n_per_class}/class → train{tuple(tx.shape)}", flush=True)
    if args.smoke:
        g = torch.Generator().manual_seed(args.seed)
        ri = torch.randperm(len(tx), generator=g)[:256]
        vi = torch.randperm(len(vx), generator=g)[:512]
        tx, ty, vx, vy = tx[ri], ty[ri], vx[vi], vy[vi]; args.epochs = 2
    model = QVFHybNoent(C).to(device)
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
    per_s = (" ".join(f"{cn[i]}={best_per[i]:.4f}" for i in range(len(cn)))
             if best_per is not None else "n/a")
    print(f"\n[DONE-QVF-HYBNOENT] {base} N={args.n_per_class or 'full'} seed={args.seed} "
          f"K={K_WIDE} best_AUC={best:.4f} per_class=({per_s}) params={nparam}", flush=True)


if __name__ == "__main__":
    main()
