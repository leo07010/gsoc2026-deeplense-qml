#!/usr/bin/env python
"""
LOW-DATA end-to-end test on raw lensing images — the one configuration not yet
cleanly tested: a TRAINABLE quantum bottleneck (CNN adapts to it) vs a
parameter-matched classical bottleneck, in the few-shot regime where a quantum
inductive-bias / regularization advantage could plausibly survive.

For each N per class and seed: train CNN→{quantum|sham}→head end-to-end on the
N-shot subsample (pathology-fixed circuit: wd=0 on angles, U(0,π) init, qlr),
evaluate on the FULL val set. Report quantum vs sham AUC, mean±std over seeds,
and a per-N win tally.  Win = quantum > sham consistently at small N.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from train_qfair import QFair                       # CNN→quantum/sham→head, P1-P4 fixed


def subsample(ty, N, C, seed):
    r = np.random.default_rng(500 + seed)
    return np.concatenate([r.choice(np.where(ty == c)[0], min(N, (ty == c).sum()),
                                    replace=False) for c in range(C)])


def auc_of(model, x, y, device, C, bs=256):
    model.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(model(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy()
    yb = label_binarize(y.numpy(), classes=np.arange(C))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def train_eval(arch, sham, tx, ty, vx, vy, C, qlr, device, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = QFair(arch, sham, C).to(device)
    qp = model.quantum_params(); qids = {id(p) for p in qp}
    base = [p for p in model.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": 1e-3, "weight_decay": 1e-4}]
    if qp:
        groups.append({"params": qp, "lr": qlr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    bs = min(64, len(tx))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=bs, shuffle=True)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        best = max(best, auc_of(model, vx, vy, device, C))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", choices=["qct", "qvf"], default="qct")
    ap.add_argument("--Ns", default="25,50,100,250,500")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qlr", type=float, default=None)
    args = ap.parse_args()
    if args.qlr is None:
        args.qlr = 1e-2 if args.arch == "qct" else 3e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]; C = len(cn)
    TX = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    TY = torch.from_numpy(d["train_y"]).long()
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1)
    vy = torch.from_numpy(d["val_y"]).long()
    Ns = [int(x) for x in args.Ns.split(",")]
    print(f"[INFO] few-shot {args.arch} qlr={args.qlr} | data={os.path.basename(args.data)} "
          f"val={len(vx)} seeds={args.seeds} Ns={Ns}", flush=True)
    print(f"  {'N/cls':>6} | {'quantum':>15} | {'sham':>15} | Δ(Q-S)  | winner")
    tally = {"Q": 0, "S": 0}
    for N in Ns:
        q, s = [], []
        for seed in range(args.seeds):
            idx = subsample(TY.numpy(), N, C, seed)
            tx, ty = TX[idx], TY[idx]
            q.append(train_eval(args.arch, False, tx, ty, vx, vy, C, args.qlr,
                                device, args.epochs, seed))
            s.append(train_eval(args.arch, True, tx, ty, vx, vy, C, args.qlr,
                                device, args.epochs, seed))
        qm, qs = np.mean(q), np.std(q)
        sm, ss = np.mean(s), np.std(s)
        win = "Q" if qm > sm else "S"; tally[win] += 1
        print(f"  {N:>6} | {qm:.4f}±{qs:.4f} | {sm:.4f}±{ss:.4f} | {qm-sm:+.4f} | {win}",
              flush=True)
    print(f"\n[DONE] {args.arch} on {os.path.basename(args.data)}: "
          f"quantum wins {tally['Q']}/{len(Ns)} N-values  "
          f"(WIN = Q wins majority AND at smallest N)")


if __name__ == "__main__":
    main()
