# GSoC 2026 — Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

> **Program:** Google Summer of Code 2026
> **Organization:** [ML4SCI](https://ml4sci.org/) — DeepLense
> **Author:** [@leo07010](https://github.com/leo07010)

Working repository for a GSoC project applying **quantum machine learning (QML)**
to strong-gravitational-lensing images, to classify the dark matter substructure
that produced them. It collects the literature review, design analysis, the
proposal draft, and the quantum-classical code that extends the current DeepLense
MAE state of the art.

> **Note on language:** This README and the structural docs are in English. The
> in-depth literature/analysis notes under [`01_Analysis/`](01_Analysis/) are
> written in **Traditional Chinese** (the author's working language).

---

## Repository layout

```
.
├── 00_Papers/              # Reference papers (PDF)
│   ├── Classical_DM_Lensing/   # Deep learning on DM lensing
│   └── Quantum_ML/             # Quantum ML techniques
├── 01_Analysis/            # Literature review & proposal (Markdown, 中文)
├── 02_Code/
│   └── mae-lensing/        # My quantum-fusion scripts + upstream analysis  ← see its README
├── 03_Data/                # Dataset location (data fetched via script, not committed)
├── download_data.py        # Fetch DeepLense datasets from Google Drive
├── setup_env.ps1           # One-shot env setup (Windows / PowerShell)
└── requirements.txt        # Python dependencies
```

> ⚠️ The ~3 GB dataset (178k `.npy` files) and the upstream
> [`achmadardanip/mae-lensing`](https://github.com/achmadardanip/mae-lensing)
> model code are **intentionally not committed**. See
> [`03_Data/README.md`](03_Data/README.md) and
> [`02_Code/mae-lensing/README.md`](02_Code/mae-lensing/README.md) for how to fetch them.

---

## Quick start

```bash
# 1. Environment
pip install -r requirements.txt          # or: ./setup_env.ps1  (Windows)

# 2. Data (~3.2 GB, downloaded into 03_Data/)
python download_data.py

# 3. Upstream model code + run (see 02_Code/mae-lensing/README.md)
git clone https://github.com/achmadardanip/mae-lensing.git
```

---

## Project idea

Replace / augment the classifier head of the SOTA DeepLense MAE model with a
**quantum fusion head**, then benchmark the hybrid against the classical baseline.

Three explored directions (full detail in
[`01_Analysis/04_GSoC_QML_Proposal.md`](01_Analysis/04_GSoC_QML_Proposal.md)):

- **D1 — Quanvolution:** 2×2 patch → small variational circuit → CNN
- **D2 — Equivariant QCNN:** symmetry-aware quantum convolution
- **D3 — Quantum MAE head:** variational circuit in place of the ViT classifier head

**Baseline to beat** (upstream MAE, mask ratio 0.9): **AUC 0.968 / accuracy 88.65%**.

---

## Analysis documents (`01_Analysis/`, 中文)

| # | Document | Contents |
|---|----------|----------|
| 01 | [QML Topic Analysis](01_Analysis/01_QML_Topic_Analysis.md) | Problem framing, why QML, technique landscape, must-read list |
| 02 | [Classical Methods (pre-2024)](01_Analysis/02_Classical_Methods_Pre2024.md) | 13 papers: CNN / ViT / equivariant / SSL / anomaly / SBI / segmentation |
| 03 | [Classical Methods (2024–2026)](01_Analysis/03_Classical_Methods_2024_2026.md) | 12 newer papers incl. current MAE SOTA |
| 04 | [GSoC QML Proposal](01_Analysis/04_GSoC_QML_Proposal.md) | Full proposal draft: directions, 12-week timeline, deliverables, risks |
| 05 | [Tesi 2024 QONN-ViT Analysis](01_Analysis/05_Tesi_2024_QONN_ViT_Analysis.md) | Quantum attention ViT for HEP |
| 06 | [Pipeline Plan](01_Analysis/06_Pipeline_Plan.md) | End-to-end implementation pipeline |

> Each `.md` has a rendered `.html` sibling for offline reading.

---

## Reference papers (`00_Papers/`)

**Classical — DeepLense baselines**
- Alexander 2019 — Deep Learning the Morphology of Dark Matter Substructure (ResNet, macro AUC 0.984)
- Alexander 2021 — Decoding Dark Matter Substructure without Supervision (AAE anomaly detection, AUC 0.932)

**Quantum ML techniques**
- Rauf et al. 2026 — Quanvolution / AstroNet
- Anwar et al. 2025 — Hybrid QC multiclass (SU(4) ansatz)
- Pasquali et al. 2024 (CERN) — Quantum Vision Transformer
- Tesi 2024 — Quantum Attention ViT for HEP

---

## Key external links

| | |
|---|---|
| ML4SCI DeepLense | <https://github.com/ML4SCI/DeepLense> |
| Upstream MAE repo | <https://github.com/achmadardanip/mae-lensing> |
| MAE paper | <https://arxiv.org/abs/2512.06642> |
| CUDA-Q | <https://nvidia.github.io/cuda-quantum/> |
| PennyLane | <https://pennylane.ai> |
| GSoC DeepLense 2025 | <https://ml4sci.org/gsoc/projects/2025/project_DEEPLENSE.html> |

---

## Status / TODO

- [x] Literature review (25+ papers)
- [x] Proposal draft v1
- [x] `QuantumFusionHead` implementation (CUDA-Q)
- [ ] Reproduce upstream baseline (AUC 0.968)
- [ ] 4-way ablation: baseline / quantum-only / concat-fusion / cross-attention fusion
- [ ] Proposal v2 (integrate fusion + ablation results)
