#!/usr/bin/env python
"""
REQAE — Rotation-Equivariant Quantum Attention Encoder (clean 2x2 test).

Motivation (from the literature survey): a quantum layer used as a better
function approximator is killed by the classical-surrogate theorem
(Schreiber-Eisert 2023) and by our measured geometric difference (g<<sqrt N).
The ONE mechanism not excluded is INDUCTIVE BIAS — and the most recent quantum
self-attention work (QPSAN 2026) attributes its gains to "structural inductive
bias, not parameter scale". Strong lensing has an approximate SO(2) symmetry and
the dark-matter class is rotation-invariant, so the natural bias is C4 rotation
invariance.

Design: a feature extractor  f = circuit∘CNN  is made C4-INVARIANT by Reynolds
twirling at the model level:
        F(x) = (1/4) Σ_{k=0..3} f( rot90(x, k) )
Both the quantum and classical arms receive the SAME group averaging, so the
2x2 comparison isolates exactly two factors:

           |   plain (no twirl)   |   equiv (C4 twirl)
  quantum  |   q-plain            |   q-equiv
  classical|   c-plain (sham)     |   c-equiv

Pre-registered reads (low-data regime, where inductive bias matters most):
  • equiv − plain > 0            : the C4 bias helps at all (sanity)
  • q-equiv − c-equiv > 0  AND  > (q-plain − c-plain)
                                  : the quantum layer contributes SPECIFICALLY
                                    under the symmetry  → genuine quantum gain
  • q-equiv ≈ c-equiv > plain     : the win is the symmetry, not the circuit
                                    (the Chang-2023 lesson)
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
from sklearn.metrics import roc_auc_score
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


class CNN(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d, 3, 2, 1), nn.BatchNorm2d(d), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class REQAE(nn.Module):
    def __init__(self, fuse, equiv, n_classes=3, d=64):
        super().__init__()
        self.fuse, self.equiv = fuse, equiv
        self.cnn = CNN(d)
        self.to_ang = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_Q))
        if fuse == "quantum":
            self.w = nn.Parameter(np.pi * torch.rand(P, N_Q, 2))
        else:                                              # classical, matched
            h = max(N_OBS, P * N_Q)
            self.core = nn.Sequential(nn.Linear(N_Q, h), nn.Tanh(),
                                      nn.Linear(h, N_OBS), nn.Tanh())
        self.head = nn.Sequential(nn.LayerNorm(N_OBS), nn.Linear(N_OBS, n_classes))

    def quantum_params(self):
        return [self.w] if self.fuse == "quantum" else []

    def _f(self, x):                                       # circuit∘CNN
        ang = torch.tanh(self.to_ang(self.cnn(x))) * np.pi
        if self.fuse == "quantum":
            return torch.stack(_circuit(ang, self.w), -1).to(ang.dtype)
        return self.core(ang)

    def forward(self, x):
        if self.equiv:                                     # C4 Reynolds twirl
            feat = sum(self._f(torch.rot90(x, k, dims=(2, 3))) for k in range(4)) / 4.0
        else:
            feat = self._f(x)
        return self.head(feat)


def auc_of(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy()
    yb = label_binarize(y.numpy(), classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def train_eval(fuse, equiv, tx, ty, vx, vy, C, device, epochs, qlr, seed, batch=128):
    torch.manual_seed(seed); np.random.seed(seed)
    model = REQAE(fuse, equiv, C).to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": 1e-3, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=min(batch, len(tx)), shuffle=True)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        best = max(best, auc_of(model, vx, vy, device, C))
    return best


def subsample(ty, N, C, seed):
    r = np.random.default_rng(500 + seed)
    return np.concatenate([r.choice(np.where(ty == c)[0], min(N, (ty == c).sum()),
                                    replace=False) for c in range(C)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--Ns", default="50,100,250,500")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qlr", type=float, default=1e-2)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    TX = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    TY = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1)
    vy = torch.from_numpy(d["val_y"]).long()
    Ns = [int(x) for x in args.Ns.split(",")]
    arms = [("quantum", True), ("classical", True), ("quantum", False), ("classical", False)]
    print(f"[INFO] REQAE | data={os.path.basename(args.data)} val={len(vx)} "
          f"seeds={args.seeds} Ns={Ns}", flush=True)
    print(f"  {'N':>5} | {'q-equiv':>8} {'c-equiv':>8} {'q-plain':>8} {'c-plain':>8} "
          f"| Δqc_equiv  Δqc_plain  Δ(eq-pl)", flush=True)
    for N in Ns:
        res = {}
        for fuse, equiv in arms:
            vals = [train_eval(fuse, equiv, TX[subsample(TY.numpy(), N, C, s)],
                               TY[subsample(TY.numpy(), N, C, s)], vx, vy, C,
                               device, args.epochs, args.qlr, s, args.batch_size)
                    for s in range(args.seeds)]
            res[(fuse, equiv)] = float(np.mean(vals))
        qe, ce = res[("quantum", True)], res[("classical", True)]
        qp, cp = res[("quantum", False)], res[("classical", False)]
        print(f"  {N:>5} | {qe:>8.4f} {ce:>8.4f} {qp:>8.4f} {cp:>8.4f} "
              f"| {qe-ce:>+8.4f}  {qp-cp:>+8.4f}  {((qe+ce)/2-(qp+cp)/2):>+8.4f}",
              flush=True)
    print("\n[VERDICT] genuine quantum gain ⟺ Δqc_equiv > 0 AND Δqc_equiv > Δqc_plain")
    print("[DONE] reqae")


if __name__ == "__main__":
    main()
