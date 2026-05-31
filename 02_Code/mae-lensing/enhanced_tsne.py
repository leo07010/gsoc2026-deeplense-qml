#!/usr/bin/env python
"""
強化版 t-SNE：把 val set 8910 張影像的 encoder features 分成 6 類標色：
  - Correct axion / Wrong axion->cdm
  - Correct cdm   / Wrong cdm->axion
  - Correct no_sub / Other errors
看誤判的 axion 是不是真的「躺進 cdm 區」。
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (
    ViTEncoder, ViTClassifier,
    build_task_VI_A_datasets,
    extract_zip_if_exists, find_dataset_root_for_classes,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", default="outputs_lens/classifier.pth")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--out_dir", default="error_analysis")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir_abs = os.path.join(repo_root, args.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    os.chdir(args.data_root)
    extract_zip_if_exists("Dataset1.zip", ".")
    dataset1_root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])

    _, _, val_ds, class_to_idx = build_task_VI_A_datasets(
        dataset1_root=dataset1_root, val_fraction=0.1, target_size=64, seed=args.seed,
    )
    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k
    print(f"[INFO] Val size: {len(val_ds)}, classes: {class_names}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    encoder = ViTEncoder(img_size=64, patch_size=4, in_chans=1,
                         embed_dim=192, depth=6, num_heads=3,
                         mlp_ratio=4.0, drop_rate=0.0)
    model = ViTClassifier(encoder, num_classes=len(class_names)).to(device)
    wp = args.weights if os.path.isabs(args.weights) \
        else os.path.join(repo_root, args.weights)
    model.load_state_dict(torch.load(wp, map_location=device))
    model.eval()
    print(f"[INFO] Loaded weights from {wp}")

    # ── Collect 192-d encoder CLS features + predictions ──
    all_feats, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits, cls_feat = model(imgs)
            preds = logits.argmax(dim=1)
            all_feats.append(cls_feat.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    print(f"[INFO] Features shape: {feats.shape}")

    # ── t-SNE ──
    print(f"[INFO] Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, random_state=args.seed,
                init="pca", learning_rate="auto",
                perplexity=args.perplexity, max_iter=1000)
    z2 = tsne.fit_transform(feats)
    print(f"[INFO] t-SNE done, shape: {z2.shape}")

    # ── Categorize each point ──
    axion_idx = class_names.index("axion")
    cdm_idx = class_names.index("cdm")
    no_sub_idx = class_names.index("no_sub")

    cat = np.full(len(labels), "other", dtype=object)
    cat[(labels == axion_idx) & (preds == axion_idx)]  = "axion_correct"
    cat[(labels == axion_idx) & (preds == cdm_idx)]    = "axion_wrong_to_cdm"
    cat[(labels == axion_idx) & (preds == no_sub_idx)] = "axion_wrong_to_nosub"
    cat[(labels == cdm_idx)   & (preds == cdm_idx)]    = "cdm_correct"
    cat[(labels == cdm_idx)   & (preds == axion_idx)]  = "cdm_wrong_to_axion"
    cat[(labels == cdm_idx)   & (preds == no_sub_idx)] = "cdm_wrong_to_nosub"
    cat[(labels == no_sub_idx) & (preds == no_sub_idx)] = "nosub_correct"
    cat[(labels == no_sub_idx) & (preds != no_sub_idx)] = "nosub_wrong"

    style_map = {
        "axion_correct":         dict(color="#1f77b4", marker="o", size=8,  alpha=0.45, label="axion correct"),
        "axion_wrong_to_cdm":    dict(color="red",     marker="X", size=30, alpha=0.95, label="axion WRONG -> cdm"),
        "axion_wrong_to_nosub":  dict(color="purple",  marker="X", size=30, alpha=0.95, label="axion WRONG -> no_sub"),
        "cdm_correct":           dict(color="#ff7f0e", marker="o", size=8,  alpha=0.45, label="cdm correct"),
        "cdm_wrong_to_axion":    dict(color="darkred", marker="^", size=30, alpha=0.95, label="cdm WRONG -> axion"),
        "cdm_wrong_to_nosub":    dict(color="brown",   marker="^", size=30, alpha=0.95, label="cdm WRONG -> no_sub"),
        "nosub_correct":         dict(color="#2ca02c", marker="o", size=8,  alpha=0.35, label="no_sub correct"),
        "nosub_wrong":           dict(color="black",   marker="s", size=30, alpha=0.95, label="no_sub WRONG"),
    }

    # ── Plot 1: full t-SNE with error highlights ──
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_order = ["nosub_correct", "axion_correct", "cdm_correct",
                  "axion_wrong_to_cdm", "cdm_wrong_to_axion",
                  "axion_wrong_to_nosub", "cdm_wrong_to_nosub", "nosub_wrong"]
    for c in plot_order:
        mask = (cat == c)
        if mask.sum() == 0:
            continue
        s = style_map[c]
        ax.scatter(z2[mask, 0], z2[mask, 1],
                   c=s["color"], marker=s["marker"], s=s["size"],
                   alpha=s["alpha"], edgecolors='none' if s["size"] < 15 else 'white',
                   linewidths=0.5,
                   label=f"{s['label']} (N={mask.sum()})")
    ax.set_title("t-SNE of encoder features\n"
                 "Large red X = axion misclassified as cdm "
                 "(does it land in cdm orange region?)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, markerscale=1.2)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    full_path = os.path.join(out_dir_abs, "tsne_with_errors.png")
    plt.savefig(full_path, dpi=140)
    plt.close()
    print(f"[INFO] Saved: {full_path}")

    # ── Plot 2: ZOOM into axion/cdm only (no no_sub) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: only correct points (clean clustering)
    for c in ["axion_correct", "cdm_correct"]:
        mask = (cat == c)
        s = style_map[c]
        axes[0].scatter(z2[mask, 0], z2[mask, 1],
                        c=s["color"], marker="o", s=10, alpha=0.5,
                        label=f"{s['label']} (N={mask.sum()})")
    axes[0].set_title("Correctly classified axion vs cdm only\n(clean clusters)")
    axes[0].legend(fontsize=10)
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    # Right: correct + wrong axion overlaid
    for c in ["axion_correct", "cdm_correct"]:
        mask = (cat == c)
        s = style_map[c]
        axes[1].scatter(z2[mask, 0], z2[mask, 1],
                        c=s["color"], marker="o", s=10, alpha=0.35,
                        label=f"{s['label']} (N={mask.sum()})")
    mask = (cat == "axion_wrong_to_cdm")
    s = style_map["axion_wrong_to_cdm"]
    axes[1].scatter(z2[mask, 0], z2[mask, 1],
                    c=s["color"], marker="X", s=40, alpha=0.95,
                    edgecolors='white', linewidths=0.7,
                    label=f"{s['label']} (N={mask.sum()})")
    axes[1].set_title("Where do the 677 misclassified axion land?\n"
                      "(Red X = WRONG; ideally they should be in BLUE region, "
                      "actually they're in ORANGE region)")
    axes[1].legend(fontsize=10)
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    zoom_path = os.path.join(out_dir_abs, "tsne_axion_vs_cdm_zoom.png")
    plt.savefig(zoom_path, dpi=140)
    plt.close()
    print(f"[INFO] Saved: {zoom_path}")

    # ── Quantitative: how far are wrong axion from cdm cluster vs axion cluster ──
    axion_correct_centroid = z2[cat == "axion_correct"].mean(axis=0)
    cdm_correct_centroid = z2[cat == "cdm_correct"].mean(axis=0)
    wrong_axion_pts = z2[cat == "axion_wrong_to_cdm"]

    d_to_axion = np.linalg.norm(wrong_axion_pts - axion_correct_centroid, axis=1)
    d_to_cdm = np.linalg.norm(wrong_axion_pts - cdm_correct_centroid, axis=1)
    closer_to_cdm = (d_to_cdm < d_to_axion).sum()

    print("\n" + "=" * 64)
    print("                 t-SNE 量化結論")
    print("=" * 64)
    print(f"Axion correct cluster centroid (t-SNE): {axion_correct_centroid}")
    print(f"CDM   correct cluster centroid (t-SNE): {cdm_correct_centroid}")
    print(f"Distance between cluster centroids:     {np.linalg.norm(axion_correct_centroid - cdm_correct_centroid):.2f}")
    print(f"\n誤判 axion (N={len(wrong_axion_pts)}) 在 t-SNE 中的位置：")
    print(f"  平均距 axion cluster: {d_to_axion.mean():.2f} ± {d_to_axion.std():.2f}")
    print(f"  平均距 cdm   cluster: {d_to_cdm.mean():.2f} ± {d_to_cdm.std():.2f}")
    print(f"  → {closer_to_cdm}/{len(wrong_axion_pts)} "
          f"({100*closer_to_cdm/len(wrong_axion_pts):.1f}%) 比較接近 CDM cluster")
    print()
    if closer_to_cdm / len(wrong_axion_pts) > 0.7:
        print("  📊 結論：誤判的 axion 在 latent space 中真的「躺進 CDM 區」")
        print("           → classical kernel 受限於這個 representation 也無解")
        print("           → 需要新的 representation (e.g., quantum) 才有救")
    elif closer_to_cdm / len(wrong_axion_pts) > 0.5:
        print("  📊 結論：多數誤判 axion 偏向 CDM 區，但部分仍在 axion 區附近")
        print("           → 部分誤判可能是 calibration 問題，非全屬 representation")
    else:
        print("  📊 結論：誤判 axion 多數仍在 axion 區附近")
        print("           → 問題出在 classifier head 而非 encoder representation")
        print("           → 換 head（例如 quantum head）可能就有救")
    print("=" * 64)


if __name__ == "__main__":
    main()
