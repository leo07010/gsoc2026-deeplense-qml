#!/usr/bin/env python
"""
QUANTUM-FAIR retraining of the end-to-end hybrids. The original runs trained
the circuit under classical defaults, which systematically handicaps it:

  P1  AdamW weight-decay on rotation ANGLES → pulls the circuit toward
      identity (angle 0 = do nothing) every step.        FIX: q-group wd=0
  P2  init 0.01·randn → circuit starts AT identity, flat & unentangled.
                                                          FIX: init U(0, π)
  P3  angle lr shared with NN weights (1e-3); PQC practice is 1e-2..1e-1.
                                                          FIX: --qlr sweep
  P4  (qvf) readout was 8 ⟨Z⟩ vs sham's full Linear(256→8) information
      access.                                FIX: readout Z + ring-ZZ = 16

Sham keeps the original (classical-optimal) settings — only the quantum
branch's pathologies are fixed. Archs: --arch qct | qvf  (+ --sham control).
Data: model_X.npz. Backend: default.qubit + backprop.
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
DIM = 2 ** N_Q
N_OBS = 2 * N_Q                                    # 8 ⟨Z⟩ + 8 ring ⟨ZZ⟩
_DEV = qml.device("default.qubit", wires=N_Q)
_SEL = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)
N_W_QCT = N_Q * 2 * N_LAYERS


def _obs():
    return [qml.expval(qml.PauliZ(j)) for j in range(N_Q)] + \
           [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _qct_circuit(angles, weights):                 # angle encode + hardware-eff. ansatz
    for i in range(N_Q):
        qml.RY(angles[..., i], wires=i)
    for L in range(N_LAYERS):
        b = L * N_Q * 2
        for i in range(N_Q):
            qml.RZ(weights[b + i * 2], wires=i)
            qml.RY(weights[b + i * 2 + 1], wires=i)
        for i in range(N_Q - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_Q - 1, 0])
    return _obs()


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _qvf_circuit(amp, weights):                    # NAE amplitudes + SEL ansatz
    qml.AmplitudeEmbedding(amp, wires=range(N_Q), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return _obs()                                  # P4 fix: 16 obs, not 8


class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class QFair(nn.Module):
    def __init__(self, arch, sham, n_classes=3, feat=128, e_hid=128):
        super().__init__()
        self.arch, self.sham = arch, sham
        self.cnn = CNNEncoder(feat)
        if arch == "qct":
            self.pre = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, N_Q))
            n_w = N_W_QCT
            sham_in = N_Q
        else:                                       # qvf
            self.pre = nn.Sequential(nn.LayerNorm(feat),
                                     nn.Linear(feat, e_hid), nn.Tanh(),
                                     nn.Linear(e_hid, DIM))
            n_w = int(np.prod(_SEL))
            sham_in = DIM
        if sham:
            self.cl = nn.Linear(sham_in, N_OBS)
        else:
            # P2 fix: init U(0, π) — proper entangling start, not identity
            self.w = nn.Parameter(np.pi * torch.rand(n_w if arch == "qct" else _SEL))
        self.head = nn.Sequential(nn.LayerNorm(N_OBS), nn.Linear(N_OBS, n_classes))

    def quantum_params(self):
        return [] if self.sham else [self.w]

    def forward(self, x):
        f = self.cnn(x)
        if self.arch == "qct":
            pre = torch.tanh(self.pre(f)) * np.pi
            z = torch.tanh(self.cl(pre)) if self.sham else \
                torch.stack(_qct_circuit(pre, self.w), -1).to(f.dtype)
        else:
            amp = torch.sqrt(torch.softmax(-self.pre(f), dim=1) + 1e-12)
            z = torch.tanh(self.cl(amp)) if self.sham else \
                torch.stack(_qvf_circuit(amp, self.w), -1).to(f.dtype)
        return self.head(z)


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
    ap.add_argument("--arch", choices=["qct", "qvf"], required=True)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qlr", type=float, default=1e-2,
                    help="circuit-angle lr (P3 fix); ignored for --sham")
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
    mode = "SHAM" if args.sham else f"QUANTUM(qlr={args.qlr})"
    print(f"[INFO] QFAIR {args.arch} {mode} | data={os.path.basename(args.data)} "
          f"train{tuple(tx.shape)} classes={cn}", flush=True)

    model = QFair(args.arch, args.sham, C).to(device)
    qp = model.quantum_params()
    qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": args.lr, "weight_decay": 1e-4}]
    if qp:                                          # P1+P3 fix: own lr, wd=0
        groups.append({"params": qp, "lr": args.qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] params={nparam} (circuit={sum(p.numel() for p in qp)})", flush=True)
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
        print(f"[ep {ep+1:02d}] AUC={m['auc']:.4f} acc={m['acc']:.4f} "
              f"f1={m['f1']:.4f} axion={axr:.4f}", flush=True)
    print(f"\n[DONE] QFAIR {args.arch} {mode} best AUC={best:.4f} params={nparam}")


if __name__ == "__main__":
    main()
