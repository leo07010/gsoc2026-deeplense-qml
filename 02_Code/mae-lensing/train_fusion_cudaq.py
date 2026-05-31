#!/usr/bin/env python
"""
STEP 2 (needs CUDA-Q — run on WSL2 / Colab / H100).

Train the F2 gated-residual quantum fusion head on the CACHED CLS features
produced by extract_features.py. The frozen ViT encoder is never touched here,
so each epoch only pays for the quantum circuit.

Design guarantees:
  * classical branch initialised from the shipped head  → start AT baseline
  * gate g = 0 at init                                  → first eval == AUC 0.974
  * only {proj, qweights, q_out, gate, classical} train → encoder stays frozen

Usage:
    python train_fusion_cudaq.py --features cls_features.npz --epochs 15
    # quick smoke test (tiny subset, verify the circuit runs & gradients flow):
    python train_fusion_cudaq.py --features cls_features.npz --smoke
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

from quantum_fusion_cudaq import QuantumFusionHead, N_Q, N_LAYERS, _HAS_CUDAQ


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
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs_quantum")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny subset to verify circuit + gradient flow.")
    args = ap.parse_args()

    if not _HAS_CUDAQ:
        raise SystemExit("[FATAL] cudaq not importable. Run on Linux/WSL/H100.")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo, args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] cudaq target may be set via CUDAQ_DEFAULT_SIMULATOR or "
          f"cudaq.set_target('nvidia'). qubits={N_Q} layers={N_LAYERS} device={device}")

    d = np.load(os.path.join(repo, args.features), allow_pickle=True)
    class_names = list(d["class_names"]); C = len(class_names)
    tf = torch.from_numpy(d["train_feats"]).float()
    tl = torch.from_numpy(d["train_labels"]).long()
    vf = torch.from_numpy(d["val_feats"]).float()
    vl = torch.from_numpy(d["val_labels"]).long()

    if args.smoke:
        tf, tl = tf[:128], tl[:128]
        vf, vl = vf[:256], vl[:256]
        args.epochs = 1
        print("[SMOKE] subset train=128 val=256, 1 epoch")

    head = QuantumFusionHead(in_dim=tf.shape[1], n_classes=C).to(device)
    head.init_classical(d["head_weight"], d["head_bias"])   # start at baseline
    print(f"[INFO] classical branch initialised from shipped head "
          f"({class_names})")

    # ── baseline check: gate=0 ⇒ should reproduce AUC 0.974 ──
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
              f"gate={np.round(gate, 3)}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(head.state_dict(), os.path.join(out_dir, "fusion_head_best.pth"))

    # ── final report focused on the axion→cdm failure mode ──
    m = evaluate(head, vf, vl, device, C)
    print("\n" + "=" * 60)
    print("                F2 FUSION RESULTS")
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
