#!/usr/bin/env python
"""
誤判分析：比較「正確分類的 axion」與「被誤判成 cdm 的 axion」之間的差異。

輸出：
  - error_analysis.csv  : 每張 val image 的 true/pred/probs + image stats
  - error_examples_grid.png : 視覺對比 axion 正確 vs 誤判 vs cdm 對照
  - confidence_distributions.png : 信心分佈
  - image_stats_comparison.png : 影像統計對比 (mean, std, max)

用法：
  cd 02_Code/mae-lensing
  python analyze_errors.py --data_root ../../03_Data
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mainv2 import (
    ViTEncoder, ViTClassifier,
    build_task_VI_A_datasets,
    extract_zip_if_exists, find_dataset_root_for_classes,
    set_seed,
)


def collect_predictions(model, loader, device):
    """Forward pass, collect logits + raw images for stat analysis."""
    model.eval()
    all_logits, all_labels, all_imgs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs_d = imgs.to(device)
            logits, _ = model(imgs_d)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_imgs.append(imgs)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    imgs = torch.cat(all_imgs).numpy()  # (N, 1, 64, 64)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    return imgs, labels, preds, probs


def image_stats(img_batch):
    """img_batch: (N, 1, H, W) -> dict of per-image stats."""
    flat = img_batch.reshape(img_batch.shape[0], -1)
    return {
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "max": flat.max(axis=1),
        "min": flat.min(axis=1),
        "p95": np.percentile(flat, 95, axis=1),
        # crude SNR proxy: peak / background-std
        "snr": flat.max(axis=1) / (flat.std(axis=1) + 1e-6),
    }


def plot_examples_grid(imgs, labels, preds, probs, class_names, save_path,
                       n_per_row=8):
    """4 rows × 8 cols: correct axion, axion-as-cdm (high conf wrong),
       axion-as-cdm (low conf wrong), correct cdm (for comparison)."""
    axion_idx = class_names.index("axion")
    cdm_idx = class_names.index("cdm")

    # Filter pools
    correct_axion = np.where((labels == axion_idx) & (preds == axion_idx))[0]
    wrong_axion_to_cdm = np.where((labels == axion_idx) & (preds == cdm_idx))[0]
    correct_cdm = np.where((labels == cdm_idx) & (preds == cdm_idx))[0]

    # Split wrong axion by model confidence in (wrong) cdm prediction
    cdm_conf = probs[wrong_axion_to_cdm, cdm_idx]
    sorted_idx = np.argsort(cdm_conf)
    low_conf_wrong = wrong_axion_to_cdm[sorted_idx[:n_per_row]]      # least confident wrong
    high_conf_wrong = wrong_axion_to_cdm[sorted_idx[-n_per_row:]]    # most confident wrong

    # Pick top-confidence correct examples
    correct_axion_conf = probs[correct_axion, axion_idx]
    correct_axion_top = correct_axion[np.argsort(correct_axion_conf)[-n_per_row:]]

    correct_cdm_conf = probs[correct_cdm, cdm_idx]
    correct_cdm_top = correct_cdm[np.argsort(correct_cdm_conf)[-n_per_row:]]

    rows = [
        ("Correct axion (high conf)", correct_axion_top, axion_idx),
        ("Wrong axion -> cdm (LOW conf, near decision boundary)", low_conf_wrong, cdm_idx),
        ("Wrong axion -> cdm (HIGH conf, confidently wrong)", high_conf_wrong, cdm_idx),
        ("Correct cdm (high conf, for comparison)", correct_cdm_top, cdm_idx),
    ]

    fig, axes = plt.subplots(len(rows), n_per_row,
                             figsize=(n_per_row * 1.8, len(rows) * 2.0))
    for r, (title, idx_arr, pred_class_idx) in enumerate(rows):
        for c in range(n_per_row):
            ax = axes[r, c]
            if c < len(idx_arr):
                i = idx_arr[c]
                ax.imshow(imgs[i, 0], cmap="viridis", vmin=0, vmax=1)
                conf = probs[i, pred_class_idx]
                ax.set_title(f"{conf:.2f}", fontsize=8)
            ax.axis("off")
        # row label on left
        axes[r, 0].set_ylabel(title, fontsize=9, rotation=90, labelpad=15)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle("Misclassification visual comparison\n"
                 "(numbers above = model confidence in shown predicted class)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_confidence_distributions(labels, preds, probs, class_names, save_path):
    axion_idx = class_names.index("axion")
    cdm_idx = class_names.index("cdm")

    correct_axion = (labels == axion_idx) & (preds == axion_idx)
    wrong_axion = (labels == axion_idx) & (preds == cdm_idx)
    correct_cdm = (labels == cdm_idx) & (preds == cdm_idx)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: confidence in "axion" prob
    axes[0].hist(probs[correct_axion, axion_idx], bins=40, alpha=0.6,
                 label=f"Correct axion (N={correct_axion.sum()})", color="C0")
    axes[0].hist(probs[wrong_axion, axion_idx], bins=40, alpha=0.6,
                 label=f"Wrong axion->cdm (N={wrong_axion.sum()})", color="C3")
    axes[0].set_xlabel("Model probability for 'axion' class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("How confident is the model that axion images ARE axion?")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: confidence in "cdm" prob
    axes[1].hist(probs[correct_cdm, cdm_idx], bins=40, alpha=0.6,
                 label=f"Correct cdm (N={correct_cdm.sum()})", color="C2")
    axes[1].hist(probs[wrong_axion, cdm_idx], bins=40, alpha=0.6,
                 label=f"Wrong axion->cdm (N={wrong_axion.sum()})", color="C3")
    axes[1].set_xlabel("Model probability for 'cdm' class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("How confident is the model when it MISTAKENLY says cdm?")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_image_stats(stats, labels, preds, class_names, save_path):
    axion_idx = class_names.index("axion")
    cdm_idx = class_names.index("cdm")

    correct_axion = (labels == axion_idx) & (preds == axion_idx)
    wrong_axion = (labels == axion_idx) & (preds == cdm_idx)
    correct_cdm = (labels == cdm_idx) & (preds == cdm_idx)

    metrics = ["mean", "std", "max", "snr"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, m in zip(axes.flatten(), metrics):
        v_corr = stats[m][correct_axion]
        v_wrong = stats[m][wrong_axion]
        v_cdm = stats[m][correct_cdm]
        ax.hist(v_corr, bins=40, alpha=0.5, label="Correct axion", color="C0", density=True)
        ax.hist(v_wrong, bins=40, alpha=0.6, label="Wrong axion->cdm", color="C3", density=True)
        ax.hist(v_cdm, bins=40, alpha=0.4, label="Correct cdm", color="C2", density=True)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.set_title(f"Image statistic: {m}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # text summary
        ax.text(0.02, 0.98,
                f"axion correct: {v_corr.mean():.3f} ± {v_corr.std():.3f}\n"
                f"axion wrong:   {v_wrong.mean():.3f} ± {v_wrong.std():.3f}\n"
                f"cdm correct:   {v_cdm.mean():.3f} ± {v_cdm.std():.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", default="outputs_lens/classifier.pth")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
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
    print(f"[INFO] Loaded weights from {wp}")

    print("[INFO] Running inference on validation set...")
    imgs, labels, preds, probs = collect_predictions(model, val_loader, device)
    stats = image_stats(imgs)
    print(f"[INFO] Got {len(imgs)} predictions")

    # CSV
    csv_path = os.path.join(out_dir_abs, "error_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label", "true_name", "pred_label", "pred_name",
                    "correct", "p_axion", "p_cdm", "p_no_sub",
                    "img_mean", "img_std", "img_max", "img_snr"])
        for i in range(len(labels)):
            w.writerow([
                i, int(labels[i]), class_names[labels[i]],
                int(preds[i]), class_names[preds[i]],
                int(labels[i] == preds[i]),
                f"{probs[i, 0]:.4f}", f"{probs[i, 1]:.4f}", f"{probs[i, 2]:.4f}",
                f"{stats['mean'][i]:.4f}", f"{stats['std'][i]:.4f}",
                f"{stats['max'][i]:.4f}", f"{stats['snr'][i]:.2f}",
            ])
    print(f"[INFO] Saved per-image CSV: {csv_path}")

    plot_examples_grid(imgs, labels, preds, probs, class_names,
                       os.path.join(out_dir_abs, "error_examples_grid.png"))
    plot_confidence_distributions(labels, preds, probs, class_names,
                                  os.path.join(out_dir_abs, "confidence_distributions.png"))
    plot_image_stats(stats, labels, preds, class_names,
                     os.path.join(out_dir_abs, "image_stats_comparison.png"))

    # Quantitative summary
    axion_idx = class_names.index("axion")
    cdm_idx = class_names.index("cdm")
    correct_axion = (labels == axion_idx) & (preds == axion_idx)
    wrong_axion = (labels == axion_idx) & (preds == cdm_idx)
    correct_cdm = (labels == cdm_idx) & (preds == cdm_idx)

    print("\n" + "=" * 64)
    print("                 KEY FINDINGS")
    print("=" * 64)
    print(f"Correct axion (axion -> axion): {correct_axion.sum()}")
    print(f"Wrong axion   (axion -> cdm):   {wrong_axion.sum()}")
    print(f"Correct cdm   (cdm -> cdm):     {correct_cdm.sum()}")

    print(f"\n--- Model confidence in (wrong) cdm prediction ---")
    cdm_conf = probs[wrong_axion, cdm_idx]
    print(f"  mean:   {cdm_conf.mean():.3f}")
    print(f"  median: {np.median(cdm_conf):.3f}")
    print(f"  >0.9:   {(cdm_conf > 0.9).sum()} / {wrong_axion.sum()} ({100*(cdm_conf > 0.9).mean():.1f}%) -- CONFIDENTLY WRONG")
    print(f"  <0.6:   {(cdm_conf < 0.6).sum()} / {wrong_axion.sum()} ({100*(cdm_conf < 0.6).mean():.1f}%) -- NEAR DECISION BOUNDARY")

    print(f"\n--- Image statistics (correct axion vs wrong axion->cdm vs correct cdm) ---")
    for m in ["mean", "std", "max", "snr"]:
        v_corr = stats[m][correct_axion]
        v_wrong = stats[m][wrong_axion]
        v_cdm = stats[m][correct_cdm]
        print(f"  {m:>4}:  correct_axion={v_corr.mean():.4f}  "
              f"wrong_axion={v_wrong.mean():.4f}  "
              f"correct_cdm={v_cdm.mean():.4f}  "
              f"(wrong vs correct_axion: {(v_wrong.mean()-v_corr.mean())/v_corr.std():+.2f} sigma)")

    print(f"\n[INFO] All artifacts in: {out_dir_abs}")
    print("=" * 64)


if __name__ == "__main__":
    main()
