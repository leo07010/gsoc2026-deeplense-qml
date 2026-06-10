#!/usr/bin/env python
"""Cache a DeepLense Model_X dataset → one .npz of TARGET×TARGET float32 images.
Handles both layouts:
  - split:  root/train/<class>/*.npy , root/val/<class>/*.npy
  - flat :  root/<class>/*.npy   (→ seeded 90/10 train/val split)
Object-array npy ([image] or [image, mass]) handled; images resized to --target."""
import os, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F


def _as2d(x):
    x = np.squeeze(np.asarray(x))                        # (1,H,W)/(H,W,1) → (H,W)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim == 3:                                      # (C,H,W) → first channel
        return np.asarray(x[0], np.float32)
    return None


def load_img(path):
    a = np.load(path, allow_pickle=True)
    if a.dtype != object:                                # plain numeric array
        r = _as2d(a)
        if r is not None:
            return r
    for e in np.asarray(a, dtype=object).reshape(-1):     # object array, e.g. [image, mass]
        r = _as2d(e)
        if r is not None:
            return r
    raise ValueError(f"no 2D image in {path} (dtype={a.dtype}, shape={a.shape})")


def collect_dir(d, classes):
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(d, c, "*.npy"))):
            xs.append(load_img(f)); ys.append(ci)
    return xs, ys


def resize_norm(imgs, target):
    """Per-image resize+normalize (images may have varying shapes)."""
    out = np.empty((len(imgs), target, target), np.float32)
    for i, im in enumerate(imgs):
        t = torch.from_numpy(np.asarray(im, np.float32))[None, None]   # (1,1,H,W)
        if t.shape[-1] != target or t.shape[-2] != target:
            t = F.interpolate(t, size=(target, target), mode="area")
        x = t[0, 0]
        mn, mx = float(x.min()), float(x.max())
        out[i] = ((x - mn) / (mx - mn + 1e-9)).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.isdir(os.path.join(args.root, "train")):
        classes = sorted(os.listdir(os.path.join(args.root, "train")))
        txs, tys = collect_dir(os.path.join(args.root, "train"), classes)
        vxs, vys = collect_dir(os.path.join(args.root, "val"), classes)
    else:                                                            # flat → split
        classes = sorted([d for d in os.listdir(args.root)
                          if os.path.isdir(os.path.join(args.root, d))])
        axs, ays = collect_dir(args.root, classes)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(axs)); nval = len(axs) // 10
        vi, ti = set(idx[:nval].tolist()), set(idx[nval:].tolist())
        txs = [axs[i] for i in range(len(axs)) if i in ti]; tys = [ays[i] for i in range(len(ays)) if i in ti]
        vxs = [axs[i] for i in range(len(axs)) if i in vi]; vys = [ays[i] for i in range(len(ays)) if i in vi]

    tx = resize_norm(txs, args.target); vx = resize_norm(vxs, args.target)
    ty = np.array(tys, np.int64); vy = np.array(vys, np.int64)
    np.savez_compressed(args.out, train_x=tx, train_y=ty, val_x=vx, val_y=vy,
                        class_names=np.array(classes))
    print(f"[INFO] {args.out}: train{tx.shape} val{vx.shape} classes={classes}", flush=True)
    print(f"[INFO] train dist {np.bincount(ty)} val {np.bincount(vy)}", flush=True)


if __name__ == "__main__":
    main()
