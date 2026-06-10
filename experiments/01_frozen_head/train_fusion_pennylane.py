#!/usr/bin/env python
"""
Train the F2 gated-residual quantum fusion head on CACHED CLS features, using
the PennyLane default.qubit + backprop backend (quantum_fusion_pennylane.py).

Same protocol as train_fusion_cudaq.py:
  * classical branch initialised from the shipped head  → start AT baseline
  * gate g = 0 at init                                  → first eval == baseline
  * only the fusion head trains (encoder already frozen + cached)

The ONLY change vs the cudaq script is the gradient engine: backprop through a
torch-resident state-vector sim (~170x faster than parameter-shift on H100),
so the full 15-epoch run takes ~1.5 h instead of ~12 days.

Usage:
    python train_fusion_pennylane.py --features cls_features.npz --epochs 15
    python train_fusion_pennylane.py --features cls_features.npz --smoke
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

_HEADS = {
    "gated": "quantum_fusion_pennylane",   # F2 gated-residual (logit-level) fusion
    "xattn": "quantum_fusion_xattn",       # cross-attention mid-fusion (arXiv:2512.19180)
}


def evaluate(head, feats, labels, device, num_classes, bs=256):
    head.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = feats[i:i + bs].to(device)
            logits.append(head(x).cpu())
    logits = torch.cat(logits)
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = probs.argmax(1)
    y = labels.numpy()
    y_bin = label_binarize(y, classes=np.arange(num_classes))
    return {
        "auc": roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"),
        "acc": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, average="macro"),
        "cm": confusion_matrix(y, y_pred),
        "y_pred": y_pred, "y": y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cls_features.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs_quantum_pl")
    ap.add_argument("--head", choices=list(_HEADS), default="gated",
                    help="gated = F2 logit-level fusion ; xattn = cross-attention mid-fusion")
    ap.add_argument("--reupload", action="store_true",
                    help="(xattn) data re-uploading: re-encode input angles every layer")
    ap.add_argument("--pure", action="store_true",
                    help="(xattn) paper-faithful pure mid-fusion: fused CLS -> logits, no gated baseline")
    ap.add_argument("--sham", action="store_true",
                    help="(xattn) ablation: quantum tokens -> classical projection (same architecture, no circuit)")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny subset to verify circuit + gradient flow.")
    args = ap.parse_args()

    # must set BEFORE importing the head module (circuit structure is bound at import)
    if args.reupload:
        os.environ["QF_REUPLOAD"] = "1"
    if args.pure:
        os.environ["QF_PURE"] = "1"
    if args.sham:
        os.environ["QF_SHAM"] = "1"
    import importlib
    mod = importlib.import_module(_HEADS[args.head])
    QuantumFusionHead, N_Q, N_LAYERS, _BACKEND = (
        mod.QuantumFusionHead, mod.N_Q, mod.N_LAYERS, mod._BACKEND)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo, args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] backend={_BACKEND}  qubits={N_Q} layers={N_LAYERS} "
          f"device={device} batch_size={args.batch_size}")

    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    class_names = list(d["class_names"]); C = len(class_names)
    tf = torch.from_numpy(d["train_feats"]).float()
    tl = torch.from_numpy(d["train_labels"]).long()
    vf = torch.from_numpy(d["val_feats"]).float()
    vl = torch.from_numpy(d["val_labels"]).long()

    if args.smoke:
        tf, tl = tf[:256], tl[:256]
        vf, vl = vf[:512], vl[:512]
        args.epochs = 1
        print("[SMOKE] subset train=256 val=512, 1 epoch")

    head = QuantumFusionHead(in_dim=tf.shape[1], n_classes=C).to(device)
    head.init_classical(d["head_weight"], d["head_bias"])   # start at baseline
    print(f"[INFO] classical branch initialised from shipped head ({class_names})")

    base = evaluate(head, vf, vl, device, C)
    print(f"[BASELINE @init] AUC={base['auc']:.4f} acc={base['acc']:.4f} "
          f"f1={base['f1']:.4f}  (gate≈0 ⇒ pure classical)")

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tf, tl), batch_size=args.batch_size, shuffle=True)

    best_auc = base["auc"]
    for ep in range(args.epochs):
        head.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(head(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        m = evaluate(head, vf, vl, device, C)
        gate = torch.tanh(head.gate).detach().cpu().numpy()
        print(f"[ep {ep+1:02d}] loss={running/len(tf):.4f}  "
              f"val AUC={m['auc']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f}  "
              f"gate={np.round(gate, 3)}", flush=True)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(head.state_dict(),
                       os.path.join(out_dir, f"fusion_head_{args.head}_best.pth"))

    m = evaluate(head, vf, vl, device, C)
    print("\n" + "=" * 60)
    print("              F2 FUSION RESULTS (PennyLane backprop)")
    print("=" * 60)
    print(f"Baseline AUC : {base['auc']:.4f}   acc {base['acc']:.4f}")
    print(f"Fusion   AUC : {m['auc']:.4f}   acc {m['acc']:.4f}  f1 {m['f1']:.4f}")
    print(f"Best val AUC : {best_auc:.4f}")
    ai = class_names.index("axion")
    axion_recall = (m["cm"][ai, ai] / m["cm"][ai].sum())
    print(f"axion recall : {axion_recall:.4f}  (baseline ~0.76 — did fusion fix it?)")
    print("Confusion (rows=true):")
    print("        " + "  ".join(f"{c:>7}" for c in class_names))
    for i, row in enumerate(m["cm"]):
        print(f"{class_names[i]:>7} " + "  ".join(f"{v:>7d}" for v in row))
    print("=" * 60)


if __name__ == "__main__":
    main()
