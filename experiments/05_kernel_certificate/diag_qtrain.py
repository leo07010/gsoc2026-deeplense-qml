#!/usr/bin/env python
"""
TRAINING DIAGNOSTIC: is the quantum circuit actually being trained, and is its
output actually used by the downstream head? Instruments a CNN→circuit→head
hybrid and logs, per epoch:

  circ_grad   : mean L2 grad norm on circuit angles    (≈0 ⇒ barren plateau / not training)
  circ_drift  : ||w − w_init||                          (≈0 ⇒ weights never moved)
  out_std     : std of circuit ⟨Z⟩ across a val batch   (≈0 ⇒ circuit outputs a constant)
  cnn_grad    : mean L2 grad norm on the CNN encoder    (does gradient flow upstream?)
  auc         : val AUC
  auc_zeroQ   : val AUC with circuit output forced to 0  (drop ⇒ head USES the circuit)

Swept over circuit learning rate qlr to check the circuit isn't merely
under/over-driven. If circ_grad>0, circ_drift grows, out_std>0, and
(auc − auc_zeroQ)>0, the circuit IS trained and IS combined — and a tie with
sham is then a real property of the task, not a training failure.
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

N_Q, N_LAYERS = 8, 4
_DEV = qml.device("default.qubit", wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(angles, weights):
    for i in range(N_Q):
        qml.RY(angles[..., i], wires=i)
    for L in range(N_LAYERS):
        for i in range(N_Q):
            qml.RZ(weights[L, i, 0], wires=i); qml.RY(weights[L, i, 1], wires=i)
        for i in range(N_Q):
            qml.CNOT(wires=[i, (i + 1) % N_Q])
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)]


class CNN(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d, 3, 2, 1), nn.BatchNorm2d(d), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.pre = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_Q))

    def forward(self, x):
        return torch.tanh(self.pre(self.net(x).flatten(1))) * np.pi   # angles


class Hybrid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.cnn = CNN()
        self.w = nn.Parameter(np.pi * torch.rand(N_LAYERS, N_Q, 2))     # U(0,π)
        self.w_init = self.w.detach().clone()
        self.head = nn.Sequential(nn.LayerNorm(N_Q), nn.Linear(N_Q, C))

    def circ_out(self, x):
        ang = self.cnn(x)
        return torch.stack(_circuit(ang, self.w), -1).to(ang.dtype)

    def forward(self, x, zero_q=False):
        z = self.circ_out(x)
        if zero_q:
            z = torch.zeros_like(z)
        return self.head(z)


def auc_of(model, x, y, device, C, zero_q=False, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device), zero_q=zero_q).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy()
    yb = label_binarize(y.numpy(), classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model_I.npz")
    ap.add_argument("--qlrs", default="1e-3,1e-2,1e-1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    rng = np.random.default_rng(0)
    ti = rng.choice(len(d["train_x"]), args.n_train, replace=False)
    vi = rng.choice(len(d["val_x"]), 3000, replace=False)
    tx = torch.from_numpy(d["train_x"][ti]).float().unsqueeze(1)
    ty = torch.from_numpy(d["train_y"][ti]).long()
    vx = torch.from_numpy(d["val_x"][vi]).float().unsqueeze(1)
    vy = torch.from_numpy(d["val_y"][vi]).long()

    for qlr in [float(x) for x in args.qlrs.split(",")]:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = Hybrid(C).to(device)
        w_init = model.w.detach().clone()
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if n != "w"],
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": [model.w], "lr": qlr, "weight_decay": 0.0}])
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        loader = DataLoader(TensorDataset(tx, ty), batch_size=256, shuffle=True)
        print(f"\n════════ qlr={qlr} ════════", flush=True)
        print(f"  {'ep':>3} | {'circ_grad':>10} | {'circ_drift':>10} | "
              f"{'out_std':>8} | {'cnn_grad':>9} | {'auc':>7} | {'auc_0Q':>7} | use", flush=True)
        for ep in range(args.epochs):
            model.train(); cg, cn_g, nb = 0.0, 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); crit(model(xb), yb).backward()
                cg += model.w.grad.norm().item()
                cn_g += torch.cat([p.grad.flatten() for p in model.cnn.parameters()
                                   if p.grad is not None]).norm().item()
                nb += 1
                opt.step()
            drift = (model.w.detach() - w_init).norm().item()
            with torch.no_grad():
                co = model.circ_out(vx[:512].to(device))
                out_std = co.std(0).mean().item()
            a = auc_of(model, vx, vy, device, C)
            a0 = auc_of(model, vx, vy, device, C, zero_q=True)
            print(f"  {ep+1:>3} | {cg/nb:>10.5f} | {drift:>10.4f} | {out_std:>8.4f} | "
                  f"{cn_g/nb:>9.3f} | {a:>7.4f} | {a0:>7.4f} | {a-a0:+.4f}", flush=True)
    print("\n[DONE] diag_qtrain")


if __name__ == "__main__":
    main()
