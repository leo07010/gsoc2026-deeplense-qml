#!/usr/bin/env python
"""
Cache the frozen-ViT PATCH tokens (B, 256, 192) for train/val, aligned 1:1 with
cls_features.npz (same seed/split/order, shuffle=False). Stored as float16 .npy
to keep size manageable (~7.9 GB train + 0.9 GB val).

These patch tokens are the classical token sequence for the Quantum-Classical
Transformer head (quantum_fusion_qct.py), which mixes them with quantum readout
tokens via self-attention.

Usage (from inside mae-lensing/):
    python extract_patch_features.py --data_root ../../03_Data
Output (repo root):
    train_patches.npy  val_patches.npy   (float16, [:,1:] patch tokens)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, argparse, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (ViTEncoder, ViTClassifier, build_task_VI_A_datasets,
                    extract_zip_if_exists, find_dataset_root_for_classes, set_seed)


@torch.no_grad()
def collect_patches(model, loader, device):
    model.eval()
    out = []
    for imgs, _ in loader:
        tok = model.encoder(imgs.to(device))      # (B, 257, 192)
        out.append(tok[:, 1:].half().cpu().numpy())  # patches (B,256,192) float16
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", default="outputs_lens/classifier.pth")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    wp = args.weights if os.path.isabs(args.weights) else os.path.join(repo, args.weights)

    os.chdir(args.data_root)
    extract_zip_if_exists("Dataset1.zip", ".")
    if os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX"); print("[INFO] Removed __MACOSX/")
    root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])
    # SAME order as cls_features: (mae_ds, train_ds, val_ds, class_to_idx), shuffle=False
    _, train_ds, val_ds, class_to_idx = build_task_VI_A_datasets(
        dataset1_root=root, val_fraction=0.1, target_size=64, seed=args.seed)
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)}")

    enc = ViTEncoder(img_size=64, patch_size=4, in_chans=1, embed_dim=192,
                     depth=6, num_heads=3, mlp_ratio=4.0, drop_rate=0.0)
    model = ViTClassifier(enc, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(wp, map_location=device))
    print(f"[INFO] loaded {wp}")

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("[INFO] extracting train patches...")
    tp = collect_patches(model, tl, device)
    print("[INFO] extracting val patches...")
    vp = collect_patches(model, vl, device)

    np.save(os.path.join(repo, "train_patches.npy"), tp)
    np.save(os.path.join(repo, "val_patches.npy"), vp)
    print(f"[INFO] saved train_patches {tp.shape} {tp.dtype}, val_patches {vp.shape}")


if __name__ == "__main__":
    main()
