# 02_Code — Quantum-classical extensions to DeepLense MAE

This folder holds **my own scripts** that extend the DeepLense MAE model with a
quantum fusion head, plus my analysis of the upstream repo. The upstream model
code is **not vendored here** — clone it and drop these scripts in.

## Setup

```bash
# 1. Clone the upstream MAE repo (provides mainv2.py, pretrained weights, data loaders)
git clone https://github.com/achmadardanip/mae-lensing.git
cd mae-lensing

# 2. Copy the scripts from this folder into the clone
#    (quantum_fusion_cudaq.py, train_fusion_cudaq.py, eval_pretrained.py, ...)

# 3. Install deps (from repo root) and fetch data
pip install -r ../../requirements.txt
python ../../download_data.py
```

The eval/feature/analysis scripts `from mainv2 import ...`, so they must sit
next to upstream's `mainv2.py`.

## My scripts

| File | Purpose | Depends on upstream |
|------|---------|:---:|
| `quantum_fusion_cudaq.py` | `QuantumFusionHead` — CUDA-Q variational circuit that replaces the classifier head (`mainv2.py:554` `ViTClassifier`) | no (standalone) |
| `train_fusion_cudaq.py` | Train the hybrid quantum-classical classifier | no (standalone) |
| `eval_pretrained.py` | Evaluate upstream's pretrained classifier (AUC / acc / confusion matrix) | yes |
| `extract_features.py` | Dump encoder features for downstream analysis | yes |
| `analyze_errors.py` | Per-class error breakdown + misclassification analysis | yes |
| `enhanced_tsne.py` | t-SNE of learned representations | yes |
| `REPO_ANALYSIS.md` | My deep-dive analysis of the upstream MAE repo (architecture, key files, integration points) | — |
| `error_analysis/` | Generated figures from `analyze_errors.py` | — |

## Upstream reference

- **Repo:** <https://github.com/achmadardanip/mae-lensing>
- **Paper:** [arXiv:2512.06642](https://arxiv.org/abs/2512.06642) — *Masked Autoencoder Pretraining on Strong-Lensing Images for Joint Dark Matter Model Classification and Super-Resolution*
- **Reported SOTA:** AUC 0.968 / accuracy 88.65% (mask ratio 0.9)

> See [`REPO_ANALYSIS.md`](REPO_ANALYSIS.md) for the full breakdown of upstream
> architecture and where the quantum branch plugs in.
