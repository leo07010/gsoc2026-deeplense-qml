#!/usr/bin/env python
"""QVF-OPT: the locked CNN+QVF hybrid architecture with the two remaining
evidence-backed optimization levers, for the low-N battlefield.

Architecture = train_qvf_hybrid.py exactly (at --nq 8 the model is
parameter-identical: Q 143,655 / sham 156,923):

    h = CNN(x) in R^128            (identical encoder, direct path to head)
    m = wide-readout(PQC(NAE(h)))  (K = 3*nq + nq*(nq-1)/2 observables)
    logits = Linear([LN(h); LN(m)])

Levers (both grounded in EXPERIMENTS_MASTER.md evidence):
  --nq {8,10,12} : qubit count. The original scratch-architecture sweep showed
      the quantum-vs-sham gap GROWS with qubits in the unsaturated regime
      (up to +0.1622 at nq=12); untested inside the hybrid architecture.
      NAE energy net stays hid=128 (faithful to the evidenced config; at nq=12
      the 128->4096 output layer dominates params — accuracy-first framing).
  --qlr : separate (higher) learning rate for the circuit angles only —
      the 96-144 quantum params share lr=1e-3 with 143k classical params by
      default; QMAE-line experiments used qlr=1e-2 for circuit params.

Battlefield: N/class in {100, 250, 500, 1000} — the only regime where the
hybrid quantum branch showed a seed-consistent gain (+2.2% at N=500 Model_II).

Protocol identical to train_qvf_scratch/wide/hybrid: AdamW 1e-3 wd=1e-4,
cosine, label_smoothing=0.05, batch 128, subsample rng 1000+seed,
best-val-AUC model selection, 20 epochs.
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

N_LAYERS = 4


def build_circuit(nq):
    dev = qml.device("default.qubit", wires=nq)
    pairs = [(i, j) for i in range(nq) for j in range(i + 1, nq)]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(amp, weights):
        qml.AmplitudeEmbedding(amp, wires=range(nq), normalize=True)
        qml.StronglyEntanglingLayers(weights, wires=range(nq))
        obs  = [qml.expval(qml.PauliZ(q)) for q in range(nq)]
        obs += [qml.expval(qml.PauliX(q)) for q in range(nq)]
        obs += [qml.expval(qml.PauliY(q)) for q in range(nq)]
        obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs]
        return obs

    return circ, 3 * nq + len(pairs)


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
    def __init__(self, in_dim, out_dim, hid=128):
        super().__init__()
        self.energy = nn.Sequential(nn.Linear(in_dim, hid), nn.Tanh(), nn.Linear(hid, out_dim))

    def forward(self, x):
        return torch.sqrt(torch.softmax(-self.energy(x), dim=1) + 1e-12)


class QVFOpt(nn.Module):
    def __init__(self, nq, n_classes=3, sham=False, feat=128):
        super().__init__()
        self.sham = sham
        self.dim = 2 ** nq
        self.circ, self.k_wide = build_circuit(nq)
        self.cnn = CNNEncoder(feat)
        self.nae = NAE(feat, self.dim)
        if sham:
            self.cl = nn.Linear(self.dim, self.k_wide)
        else:
            shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=nq)
            self.w = nn.Parameter(0.1 * torch.randn(shape))
        self.ln_h = nn.LayerNorm(feat)
        self.ln_m = nn.LayerNorm(self.k_wide)
        self.head = nn.Linear(feat + self.k_wide, n_classes)

    # ------------------------------------------------------------------
    # KNOWN BUG (disclosed, NOT fixed — this file is preserved exactly as
    # it produced the disputed "+1.4% quantum contribution" claim; see
    # docs/BENCHMARK_DATABASE.csv row id=29 for the full corrected verdict).
    #
    # `amp = self.nae(h)` below runs UNCONDITIONALLY, before the
    # `if self.sham:` branch. That means the sham ("classical control") arm
    # still pays the full NAE energy-net bottleneck cost (Linear->Tanh->Linear
    # projecting h down to a `dim`-way Boltzmann amplitude vector) and only
    # swaps the quantum circuit for `self.cl` (a Linear layer) downstream of
    # it. The sham was intended to isolate "quantum circuit vs. no quantum
    # circuit, everything else identical", but because it still shares the
    # NAE bottleneck, it does NOT reduce to the true classical floor (the
    # plain CNN in qvf_plain_baseline.py with no bottleneck at all).
    #
    # Consequence: the independent adversarial statistical audit
    # (2026-07-31, see BENCHMARK_DATABASE.csv row 29) found the sham
    # underperforms plain-lin by -0.006 (p=.885) — i.e. the NAE bottleneck
    # itself is a real handicap the sham unfairly carries. That means part
    # of the reported "+1.4% quantum contribution" (Q vs. sham delta) is
    # actually "no-bottleneck vs. bottleneck", not "quantum vs. classical".
    # The audit estimates ~42% of the reported effect is attributable to
    # this broken control, not a genuine quantum effect.
    #
    # Correct fix (NOT applied here, to preserve this file as-run): the sham
    # arm should skip `self.nae()` entirely and route `h` directly to a
    # classical projection matched in output width to `m` (mirroring how
    # zeroing/bypassing the quantum branch gives the plain-CNN floor
    # elsewhere in this architecture family, e.g. train_qvf_hybrid.py's
    # "zeroing the m block reduces exactly to plain-lin" guarantee). Do not
    # rerun this script for a quantum-vs-classical comparison and reuse the
    # sham numbers without applying that fix first.
    # ------------------------------------------------------------------
    def forward(self, x):
        h = self.cnn(x)
        amp = self.nae(h)
        if self.sham:
            m = torch.tanh(self.cl(amp))
        else:
            m = torch.stack(self.circ(amp, self.w), dim=-1).to(amp.dtype)
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
    ap.add_argument("--nq", type=int, default=8, choices=[8, 10, 12])
    ap.add_argument("--qlr", type=float, default=0.0, help="separate lr for circuit angles (0 = shared)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    ty = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1); vy = torch.from_numpy(d["val_y"]).long()
    mode = "SHAM" if args.sham else "QUANTUM"
    print(f"[INFO] QVF-OPT {mode} nq={args.nq} qlr={args.qlr} | data={os.path.basename(args.data)} "
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
    model = QVFOpt(args.nq, C, sham=args.sham).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={nparam} (K_wide={model.k_wide})")
    if args.qlr > 0 and not args.sham:
        rest = [p for n_, p in model.named_parameters() if n_ != "w"]
        opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                                 {"params": [model.w], "lr": args.qlr}], weight_decay=1e-4)
    else:
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
    print(f"\n[DONE-QVF-OPT] {mode} nq={args.nq} qlr={args.qlr} {base} N={args.n_per_class or 'full'} "
          f"seed={args.seed} best_AUC={best:.4f} per_class=({per_s}) params={nparam}", flush=True)


if __name__ == "__main__":
    main()
