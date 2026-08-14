#!/usr/bin/env python3
"""Step-0 diagnostic for the token-space quantum mixer proposal
(2026-08-14): do the three classes actually separate in spatial power
spectrum? If yes, a Fourier/multiscale-structured token mixer (the
RY+CNOT circuit acting on the patch-index axis) is a data-justified
inductive bias; if no, that proposal is downgraded before any
architecture work is spent on it.

Pure numpy diagnostic on a subsample -- no training, no GPU.

For each dataset (Model_II, Model_III) and each class:
  1. Radially-averaged 2D FFT power spectrum, mean +/- std over images.
  2. Pairwise class separability per radial frequency bin:
     Cohen's d = |mu_a - mu_b| / pooled_std  (on log-power).
  3. A patch-level 8x8 "bit-plane" energy map: project each image onto
     the 64-patch grid the ViT uses (8x8 patches of 8x8 pixels), FFT the
     64-dim patch-mean vector along the token index, and report
     class-wise energy per token-frequency -- this is exactly the axis
     the proposed quantum mixer would act on.

Output: one JSON with the numbers + one PNG figure per dataset.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def radial_power_spectrum(imgs):
    """imgs: (n, H, W) float. Returns (n, n_bins) radially-averaged
    log10 power spectra."""
    n, H, W = imgs.shape
    f = np.fft.fftshift(np.fft.fft2(imgs), axes=(-2, -1))
    p = np.abs(f) ** 2
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    n_bins = r.max() + 1
    out = np.zeros((n, n_bins))
    counts = np.bincount(r.ravel(), minlength=n_bins)
    for i in range(n):
        out[i] = np.bincount(r.ravel(), weights=p[i].ravel(),
                             minlength=n_bins) / np.maximum(counts, 1)
    return np.log10(out + 1e-12)

def cohens_d(a, b):
    """Per-column Cohen's d between two (n, k) arrays."""
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a.std(0, ddof=1), b.std(0, ddof=1)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return np.abs(ma - mb) / np.maximum(pooled, 1e-12)

def patch_token_spectrum(imgs, patch=8):
    """Mean-pool to the 8x8 patch grid (the ViT token grid), flatten to a
    64-dim token vector, FFT along the token index -> log-power per
    token-frequency. This is the axis the proposed quantum mixer acts on."""
    n, H, W = imgs.shape
    g = imgs.reshape(n, H // patch, patch, W // patch, patch).mean(axis=(2, 4))
    tok = g.reshape(n, -1)                       # (n, 64), row-major
    f = np.fft.fft(tok - tok.mean(1, keepdims=True), axis=1)
    return np.log10(np.abs(f[:, :tok.shape[1] // 2]) ** 2 + 1e-12)

def analyze(path, name, n_per_class, out_prefix):
    d = np.load(path, allow_pickle=True)
    cn = [str(c) for c in d["class_names"]]
    x, y = d["train_x"], d["train_y"]
    if x.ndim == 4:            # (n,1,H,W) -> (n,H,W)
        x = x[:, 0]
    rng = np.random.default_rng(0)
    spectra, tokspec = {}, {}
    for c, cname in enumerate(cn):
        idx = np.where(y == c)[0]
        idx = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        imgs = x[idx].astype(np.float64)
        spectra[cname] = radial_power_spectrum(imgs)
        tokspec[cname] = patch_token_spectrum(imgs)

    # pairwise separability
    pairs = [(a, b) for i, a in enumerate(cn) for b in cn[i + 1:]]
    report = {"dataset": name, "classes": cn, "n_per_class": n_per_class,
              "radial": {}, "token_axis": {}}
    for a, b in pairs:
        d_rad = cohens_d(spectra[a], spectra[b])
        d_tok = cohens_d(tokspec[a], tokspec[b])
        report["radial"][f"{a}_vs_{b}"] = {
            "max_d": float(d_rad.max()),
            "argmax_bin": int(d_rad.argmax()),
            "n_bins_d_gt_0.8": int((d_rad > 0.8).sum()),
            "n_bins_d_gt_0.5": int((d_rad > 0.5).sum()),
            "n_bins": int(len(d_rad)),
        }
        report["token_axis"][f"{a}_vs_{b}"] = {
            "max_d": float(d_tok.max()),
            "argmax_freq": int(d_tok.argmax()),
            "n_freqs_d_gt_0.8": int((d_tok > 0.8).sum()),
            "n_freqs": int(len(d_tok)),
        }

    # figure: spectra + separability
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for cname in cn:
        m, s = spectra[cname].mean(0), spectra[cname].std(0)
        axes[0].plot(m, label=cname)
        axes[0].fill_between(range(len(m)), m - s, m + s, alpha=0.15)
    axes[0].set_title(f"{name}: radial log-power (mean±std)")
    axes[0].set_xlabel("radial frequency bin"); axes[0].legend()
    for a, b in pairs:
        axes[1].plot(cohens_d(spectra[a], spectra[b]), label=f"{a} vs {b}")
    axes[1].axhline(0.8, ls=":", c="gray"); axes[1].axhline(0.5, ls=":", c="lightgray")
    axes[1].set_title("radial separability (Cohen's d, log-power)")
    axes[1].set_xlabel("radial frequency bin"); axes[1].legend()
    for a, b in pairs:
        axes[2].plot(cohens_d(tokspec[a], tokspec[b]), label=f"{a} vs {b}")
    axes[2].axhline(0.8, ls=":", c="gray")
    axes[2].set_title("token-axis (8x8 patch grid) separability")
    axes[2].set_xlabel("token-index frequency"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_{name}.png", dpi=150)
    plt.close(fig)
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=300)
    ap.add_argument("--out_prefix", default="spectral_diag")
    args = ap.parse_args()
    reports = []
    for name, path in [("model_II", "/home/leo07010/mae-lensing/model_II.npz"),
                       ("model_III", "/home/leo07010/mae-lensing/model_III.npz")]:
        r = analyze(path, name, args.n_per_class, args.out_prefix)
        reports.append(r)
        print(json.dumps(r, indent=1))
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(reports, f, indent=1)
    print("done")
