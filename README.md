# Quantum ML × DeepLense — Dark-Matter Substructure from Strong Lensing

Systematic, **sham-controlled** study of hybrid quantum-classical machine learning on the
[ML4SCI DeepLense](https://github.com/ML4SCI/DeepLense) dark-matter benchmark
(3-class: `axion` vortex / `cdm` subhalos / `no_sub`), built on top of the classical
MAE SOTA ([arXiv:2512.06642](https://arxiv.org/abs/2512.06642), AUC 0.968).

**Current proposal:** [PROPOSAL.md](PROPOSAL.md) — *Quantum-Compressed Anomaly Detection of
Dark-Matter Substructure on Self-Supervised Lensing Representations*
**All measured numbers:** [docs/RESULTS.md](docs/RESULTS.md)

## Key findings so far

1. **Sham controls change the story.** Every quantum model here is paired with a
   capacity-matched classical control ("sham"). On frozen discriminative features, four
   different quantum architectures all *exactly tie* their shams (AUC ≈ 0.98) — published
   hybrid-QML gains without such controls are likely classical-wrapper effects.
2. **The quantum-vs-classical question is training-regime dependent.** Quantum-over-sham gaps
   (+0.007–0.010 AUC, concentrated on the hardest class, axion) appear only when gradients
   flow through the circuit into the representation (end-to-end / pretrain-finetune), never
   on frozen features. The MAE paper's own frozen-probe result (AUC 0.5365) explains why.
3. **Strongest quantum result:** a 72-parameter trash-qubit quantum autoencoder matches a
   2,308-parameter classical AE at substructure-anomaly AUC ≈ 0.996 — now being re-validated
   on leakage-free self-supervised features with a parameter-matched control
   (`experiments/04_qae_ensemble/`).

## Method taxonomy

| Category | Question | Scripts | Status |
|---|---|---|---|
| `experiments/00_baselines/` | classical reference, error analysis, feature caches | `eval_pretrained` `analyze_errors` `extract_features*` `cache_model` `classical_control` | ✅ measured |
| `experiments/01_frozen_head/` | quantum heads on frozen features (gated / x-attn / QCT / QVF) | `train_fusion_*` `train_qct` `train_qvf_cls` | ✅ quantum = sham |
| `experiments/02_generative_ssl/` | QMAE, QAE anomaly, equivariant, few-shot | `train_qmae*` `train_qae_anomaly*` `train_fewshot` | ✅ measured |
| `experiments/03_end_to_end/` | circuit shapes the representation (scratch / pretrain→finetune) | `train_*_scratch` `pretrain_finetune` | ✅ quantum > sham (single seed) |
| `experiments/04_qae_ensemble/` | **class-conditional QAE ensemble: anomaly + generative 3-class + open-set discovery** | `train_qae_ensemble` | 🕐 running |
| `models/` | quantum circuits & hybrid architectures (PennyLane) | `quantum_*` | — |

## Repository layout

```
├── PROPOSAL.md              research proposal (current)
├── docs/
│   ├── RESULTS.md           consolidated measured results ⭐
│   └── analysis/            literature surveys, upstream-repo dissection, designs
├── models/                  quantum circuit / hybrid architecture modules
├── experiments/             training & evaluation scripts, by method category
├── slurm/                   HPC job scripts (sbatch + drivers)
├── results/                 committed artifacts (error analysis, result CSVs)
├── papers/                  key reference PDFs
├── data/                    dataset download instructions (data not committed)
└── download_data.py
```

## Setup

```bash
pip install -r requirements.txt
python download_data.py            # DeepLense Dataset1/2 → 03_Data/
```

The MAE upstream code (`mainv2.py`, no license published) is **not** vendored; fetch it from
[achmadardanip/mae-lensing](https://github.com/achmadardanip/mae-lensing) into your working
directory. Scripts import sibling modules from `models/` — run with
`PYTHONPATH=<repo>/models` or from a flat working directory.

Typical run (see `slurm/`):

```bash
# leakage-free features from the self-supervised encoder
python experiments/00_baselines/extract_features_ssl.py --data model_I.npz --encoder enc_I.pth
# QAE ensemble, all four arms
python experiments/04_qae_ensemble/train_qae_ensemble.py --arm quantum --seeds 42
```

## Honesty rules of this repo

- Every quantum number ships with its sham control.
- Single-seed results are labelled as such; incomplete runs are labelled *incomplete*.
- Negative results are reported, not buried (see Regime A ties, Model_IV data bug).
