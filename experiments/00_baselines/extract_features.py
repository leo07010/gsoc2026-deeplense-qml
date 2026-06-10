#!/usr/bin/env python
"""
STEP 1 (runs on native Windows, torch-only — no CUDA-Q needed).

Freeze the shipped MAE ViT encoder and cache the 192-d CLS features for the
train/val splits to a single .npz. The quantum fusion head is then trained on
these cached features (STEP 2, train_fusion_cudaq.py) WITHOUT re-running the ViT
every epoch — the quantum circuit is the bottleneck, so we pay the encoder cost
exactly once.

Also stores the shipped classifier head weight/bias so the fusion head's
classical branch can be initialised to the EXACT baseline (AUC 0.974).

Usage (from inside mae-lensing/):
    python extract_features.py --data_root ../../03_Data
Output:
    cls_features.npz  (train_feats, train_labels, val_feats, val_labels,
                       head_weight, head_bias, class_names)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP clash workaround

import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (
    ViTEncoder, ViTClassifier,
    build_task_VI_A_datasets,
    extract_zip_if_exists, find_dataset_root_for_classes,
    set_seed,
)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        _, cls_feat = model(imgs)          # (B, 192)
        feats.append(cls_feat.cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", default="outputs_lens/classifier.pth")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="cls_features.npz")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(repo_root, args.out)
    wp = args.weights if os.path.isabs(args.weights) \
        else os.path.join(repo_root, args.weights)

    os.chdir(args.data_root)
    extract_zip_if_exists("Dataset1.zip", ".")
    # macOS-zipped archives ship a __MACOSX/ tree of AppleDouble `._*.npy` files.
    # find_dataset_root_for_classes() can mis-select __MACOSX/Dataset as the data
    # root (filesystem-order dependent), poisoning every sample with random noise.
    # extract_zip_if_exists() re-extracts every run, so clean it every run.
    import shutil
    if os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX")
        print("[INFO] Removed __MACOSX/ (macOS zip metadata)")
    dataset1_root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])

    # build_task_VI_A_datasets returns (mae_ds, train_ds, val_ds, class_to_idx)
    _, train_ds, val_ds, class_to_idx = build_task_VI_A_datasets(
        dataset1_root=dataset1_root, val_fraction=0.1, target_size=64, seed=args.seed,
    )
    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}  classes={class_names}")

    encoder = ViTEncoder(img_size=64, patch_size=4, in_chans=1,
                         embed_dim=192, depth=6, num_heads=3,
                         mlp_ratio=4.0, drop_rate=0.0)
    model = ViTClassifier(encoder, num_classes=len(class_names)).to(device)
    state = torch.load(wp, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded shipped weights: {wp}")

    head_w = state["head.weight"].cpu().numpy()   # (3, 192) — baseline classical head
    head_b = state["head.bias"].cpu().numpy()     # (3,)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    vl = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("[INFO] Extracting train features...")
    tf, tlab = collect(model, tl, device)
    print("[INFO] Extracting val features...")
    vf, vlab = collect(model, vl, device)

    np.savez_compressed(
        out_path,
        train_feats=tf.astype(np.float32), train_labels=tlab.astype(np.int64),
        val_feats=vf.astype(np.float32),   val_labels=vlab.astype(np.int64),
        head_weight=head_w.astype(np.float32), head_bias=head_b.astype(np.float32),
        class_names=np.array(class_names),
    )
    print(f"[INFO] Saved {out_path}")
    print(f"       train_feats {tf.shape}  val_feats {vf.shape}")


if __name__ == "__main__":
    main()
