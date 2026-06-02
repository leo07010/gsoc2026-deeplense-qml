#!/usr/bin/env python
"""Downsample DeepLense lensing images to 16x16 and cache a subset for QMAE.
Output: img16.npz (train_x, train_y, val_x, val_y, class_names)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, argparse, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (build_task_VI_A_datasets, extract_zip_if_exists,
                    find_dataset_root_for_classes, set_seed)


def collect(ds, n):
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    xs, ys, got = [], [], 0
    for img, y in loader:                       # img (B,1,16,16)
        xs.append(img[:, 0].numpy()); ys.append(y.numpy()); got += len(y)
        if got >= n:
            break
    x = np.concatenate(xs)[:n]; y = np.concatenate(ys)[:n]
    return x.astype(np.float32), y.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    repo = os.path.dirname(os.path.abspath(__file__))
    os.chdir(args.data_root)
    extract_zip_if_exists("Dataset1.zip", ".")
    if os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX")
    root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])
    _, train_ds, val_ds, c2i = build_task_VI_A_datasets(
        dataset1_root=root, val_fraction=0.1, target_size=16, seed=args.seed)  # 16x16!
    cn = [None] * len(c2i)
    for k, v in c2i.items():
        cn[v] = k
    tx, ty = collect(train_ds, args.n_train)
    vx, vy = collect(val_ds, args.n_val)
    out = os.path.join(repo, "img16.npz")
    np.savez_compressed(out, train_x=tx, train_y=ty, val_x=vx, val_y=vy,
                        class_names=np.array(cn))
    print(f"[INFO] saved {out}  train_x{tx.shape} val_x{vx.shape} classes={cn}")
    print(f"[INFO] train label dist: {np.bincount(ty)}  val: {np.bincount(vy)}")


if __name__ == "__main__":
    main()
