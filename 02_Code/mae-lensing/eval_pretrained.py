#!/usr/bin/env python
"""
Load shipped classifier.pth and evaluate on validation split.
Run from inside mae-lensing/ directory:
    python eval_pretrained.py --data_root ../../03_Data

No retraining. Reports: AUC, accuracy, per-class P/R/F1, confusion matrix.
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (
    ViTEncoder, ViTClassifier,
    build_task_VI_A_datasets,
    extract_zip_if_exists, find_dataset_root_for_classes,
    set_seed,
)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    y_bin = label_binarize(labels, classes=np.arange(num_classes))
    return {
        "auc": roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"),
        "acc": accuracy_score(labels, y_pred),
        "f1": f1_score(labels, y_pred, average="macro"),
        "cm": confusion_matrix(labels, y_pred),
        "report": classification_report(labels, y_pred, digits=4),
        "labels": labels,
        "y_pred": y_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder containing Dataset1.zip OR already-extracted axion/cdm/no_sub/")
    ap.add_argument("--weights", default="outputs_lens/classifier.pth",
                    help="Path to shipped classifier.pth")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Resolve data root
    os.chdir(args.data_root)
    extract_zip_if_exists("Dataset1.zip", ".")
    dataset1_root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])
    if not dataset1_root:
        raise RuntimeError("Dataset1 not found under " + args.data_root)

    _, _, val_ds, class_to_idx = build_task_VI_A_datasets(
        dataset1_root=dataset1_root, val_fraction=0.1, target_size=64, seed=args.seed,
    )
    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k
    print(f"[INFO] Validation size: {len(val_ds)}, classes: {class_names}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Build & load
    encoder = ViTEncoder(img_size=64, patch_size=4, in_chans=1,
                         embed_dim=192, depth=6, num_heads=3,
                         mlp_ratio=4.0, drop_rate=0.0)
    model = ViTClassifier(encoder, num_classes=len(class_names)).to(device)

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        # Resolve relative to repo root, not data_root
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), weights_path
        )
    print(f"[INFO] Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    print("[INFO] Weights loaded.")

    metrics = evaluate(model, val_loader, device, num_classes=len(class_names))

    print("\n" + "=" * 60)
    print("                EVALUATION RESULTS")
    print("=" * 60)
    print(f"Macro AUC : {metrics['auc']:.4f}")
    print(f"Accuracy  : {metrics['acc']:.4f}")
    print(f"Macro F1  : {metrics['f1']:.4f}")
    print("\nClassification report:\n" + metrics["report"])
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c:>8}" for c in class_names))
    for i, row in enumerate(metrics["cm"]):
        print(f"  {class_names[i]:>8} | " + "  ".join(f"{v:>8d}" for v in row))
    print("=" * 60)


if __name__ == "__main__":
    main()
