# Quantum ML × DeepLense — Dark-Matter Substructure from Strong Lensing

Systematic, **sham-controlled** study of hybrid quantum-classical machine learning on the
[ML4SCI DeepLense](https://github.com/ML4SCI/DeepLense) dark-matter benchmark
(3-class: `axion` vortex / `cdm` subhalos / `no_sub`), built on top of the classical
MAE SOTA ([arXiv:2512.06642](https://arxiv.org/abs/2512.06642), AUC 0.968).

**Benchmark database (start here):** [docs/BENCHMARK.md](docs/BENCHMARK.md) +
[docs/BENCHMARK_DATABASE.csv](docs/BENCHMARK_DATABASE.csv) — a structured, one-row-per-method
table of all 32 quantum-vs-classical experiments run in this project, with matched-control
deltas, seed counts, and an explicit verdict per row. This is the current, corrected,
authoritative summary; the sections below and [docs/RESULTS.md](docs/RESULTS.md) /
[docs/EXPERIMENTS_MASTER.md](docs/EXPERIMENTS_MASTER.md) are the earlier narrative logs it was
built from.
**Original proposal (historical):** [PROPOSAL.md](PROPOSAL.md) — predates the QVF-Hybrid
campaign below; kept for the project's narrative record, see its status note.

## Key findings so far

1. **Sham controls change the story.** Every quantum model here is paired with a
   capacity-matched classical control ("sham"). On frozen discriminative features, four
   different quantum architectures all *exactly tie* their shams (AUC ≈ 0.98) — published
   hybrid-QML gains without such controls are likely classical-wrapper effects.
2. **Across ~28 architectures tried on image classification** (fusion heads, quanvolution,
   quantum kernels, generative/anomaly models, equivariant layers, attention variants — see
   the benchmark database): 6 beat their matched sham, 10 tied, 7 lost, 2 were proven
   *certificate-impossible* (no quantum-kernel advantage can exist on this data, for any
   labeling), 4 are inconclusive, and 2 were retracted after closer scrutiny.
3. **2026-07-31 self-correction.** This project's own strongest-looking result — a residual
   quantum branch ("QVF-Hybrid") reported as a statistically significant module-level effect
   (p=0.0032) — did **not** survive an independent adversarial audit: the sham control had a
   bug that kept it from reducing to the true classical floor, the p-value fails
   multiple-comparisons correction by ~90× once the full search that found this cell is
   accounted for, and the held-out confirmatory seeds alone are null. Verdict corrected to
   `retracted`. Full reasoning in [docs/BENCHMARK_DATABASE.csv](docs/BENCHMARK_DATABASE.csv)
   row 29 and [docs/BENCHMARK.md](docs/BENCHMARK.md). The code that produced it — bug included,
   disclosed inline — is published as-is in
   [`experiments/07_qvf_final_locked/`](experiments/07_qvf_final_locked/).
4. **The one advantage that does still stand** is not an image-classification result: a
   theorem-backed quantum-memory advantage for quantum-sensor metrology (row 32 in the
   benchmark database) — a different task from DeepLense classification, kept in its own
   category so it's never conflated with the (retracted) classification claim.
5. **Earlier result, superseded by the leakage-free re-run:** a 72-parameter trash-qubit
   quantum autoencoder once matched a 2,308-parameter classical AE at anomaly AUC ≈ 0.996 on
   label-fine-tuned features; re-run on leakage-free self-supervised features
   (`experiments/04_qae_ensemble/`) it collapsed to AUC ≈ 0.44 — the leakage warning is
   documented as its own row (id 13) in the benchmark database.

## Method taxonomy

| Category | Question | Scripts | Status |
|---|---|---|---|
| `experiments/00_baselines/` | classical reference, error analysis, feature caches | `eval_pretrained` `analyze_errors` `extract_features*` `cache_model` `classical_control` | ✅ measured |
| `experiments/01_frozen_head/` | quantum heads on frozen features (gated / x-attn / QCT / QVF) | `train_fusion_*` `train_qct` `train_qvf_cls` | ✅ quantum = sham |
| `experiments/02_generative_ssl/` | QMAE, QAE anomaly, equivariant, few-shot | `train_qmae*` `train_qae_anomaly*` `train_fewshot` | ✅ measured |
| `experiments/03_end_to_end/` | circuit shapes the representation (scratch / pretrain→finetune) | `train_*_scratch` `pretrain_finetune` | ✅ quantum > sham (single seed) |
| `experiments/04_qae_ensemble/` | **class-conditional QAE ensemble: anomaly + generative 3-class + open-set discovery** | `train_qae_ensemble` | 🕐 running |
| `experiments/07_qvf_final_locked/` | the final QVF-Hybrid residual-branch architecture (incl. the null-baseline discovery and entanglement ablation) — the code that produced the retracted claim, bug disclosed inline | `qvf_plain_baseline` `train_qvf_scratch` `train_qvf_hybrid` `train_qvf_opt` `train_qvf_hybnoent` `train_qvf_noent` | ⚠️ retracted, see docs/BENCHMARK_DATABASE.csv row 29 |
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
