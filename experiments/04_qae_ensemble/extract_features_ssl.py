#!/usr/bin/env python
"""
Extract LEAKAGE-FREE 192-d CLS features from a cached Model_X dataset using a
self-supervised MAE encoder (pretrained on no_sub reconstruction ONLY — no
labels anywhere). This replaces cls_features.npz, whose encoder was fine-tuned
with 3-class labels (label leakage → invalid for unsupervised anomaly claims).

Usage:
    python extract_features_ssl.py --data model_I.npz --encoder enc_I.pth \
                                   --out ssl_features_I.npz
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch

from mainv2 import ViTEncoder


def make_encoder():
    return ViTEncoder(img_size=64, patch_size=4, in_chans=1, embed_dim=192,
                      depth=6, num_heads=3, mlp_ratio=4.0, drop_rate=0.0)


@torch.no_grad()
def extract(enc, x, device, bs=256):
    feats = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        feats.append(enc(xb)[:, 0].cpu())            # CLS token (B,192)
    return torch.cat(feats).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model_I.npz")
    ap.add_argument("--encoder", default="enc_I.pth")
    ap.add_argument("--out", default="ssl_features_I.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(repo, args.data), allow_pickle=True)
    tx = torch.from_numpy(d["train_x"]).float().unsqueeze(1)
    vx = torch.from_numpy(d["val_x"]).float().unsqueeze(1)

    enc = make_encoder()
    enc.load_state_dict(torch.load(os.path.join(repo, args.encoder), map_location=device))
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    print(f"[INFO] SSL encoder={args.encoder} data={args.data} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)}", flush=True)
    tf = extract(enc, tx, device)
    vf = extract(enc, vx, device)
    out = os.path.join(repo, args.out)
    np.savez_compressed(out, train_feats=tf, train_labels=d["train_y"],
                        val_feats=vf, val_labels=d["val_y"],
                        class_names=d["class_names"])
    print(f"[INFO] saved {out}: train_feats{tf.shape} val_feats{vf.shape} "
          f"classes={list(d['class_names'])}", flush=True)


if __name__ == "__main__":
    main()
