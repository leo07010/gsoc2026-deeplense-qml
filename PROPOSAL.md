# Research Proposal

## Quantum-Compressed Anomaly Detection of Dark-Matter Substructure on Self-Supervised Lensing Representations

> **Author:** [@leo07010](https://github.com/leo07010) · leo07010@gmail.com
> **Target program:** GSoC 2026 — ML4SCI / DeepLense (QML track)
> **Status:** experiments in progress — see [docs/RESULTS.md](docs/RESULTS.md) for all measured numbers
> **Date:** 2026-06-10

---

## Abstract

Strong gravitational lensing encodes dark-matter (DM) substructure signatures — CDM point-like
subhalos and axion vortex lines — as sub-percent perturbations of lensing arcs. The classical
state of the art on the DeepLense 3-class benchmark is an MAE-pretrained ViT
(AUC 0.968, [arXiv:2512.06642](https://arxiv.org/abs/2512.06642)). We conducted a systematic,
**sham-controlled** study of hybrid quantum-classical models on this task and found that the
quantum-vs-classical question is **training-regime dependent**: quantum heads bolted onto frozen
discriminative features never differ from capacity-matched classical controls (the feature
ceiling is saturated), while small but consistent quantum-over-sham gaps appear only when
gradients flow through the circuit into the representation. Building on the strongest observed
quantum result — a 72-parameter trash-qubit quantum autoencoder matching a 2,308-parameter
classical control at anomaly AUC ≈ 0.996 — we propose a **class-conditional quantum autoencoder
ensemble** on leakage-free self-supervised features that performs (i) label-free substructure
anomaly detection, (ii) generative 3-class classification via calibrated reconstruction
fidelity, and (iii) **open-set discovery of dark-matter models never seen in training** —
a capability no discriminative DeepLense classifier has, and one directly relevant to real
surveys where the true DM model may be none of the simulated ones.

---

## 1. Background and gap

| Fact | Source |
|---|---|
| DeepLense classical SOTA: MAE-pretrained ViT, AUC 0.968 / acc 88.65% | arXiv:2512.06642 |
| Classical bottleneck: axion recall ≈ 0.77, dominated by axion↔cdm confusion | our error analysis ([results/error_analysis](results/error_analysis)) |
| Unsupervised DeepLense line: adversarial autoencoder, anomaly AUC ≈ 0.93 | Alexander et al. 2021 |
| **No quantum work exists in the DeepLense ecosystem (2019–2026)** | literature survey ([docs/analysis](docs/analysis)) |
| Most published hybrid-QML gains lack capacity-matched classical controls | motivates our sham methodology |

## 2. What our systematic study established

Every quantum experiment in this repo is paired with a **sham control** — an identical
architecture in which the quantum circuit is replaced by a classical projection of matched
dimensionality. Full numbers in [docs/RESULTS.md](docs/RESULTS.md); the structure of the
findings:

| Training regime | Quantum − Sham | Interpretation |
|---|---|---|
| **A. Frozen fine-tuned features** (4 architectures) | ≈ 0 (all tie at AUC ≈ 0.98) | features already saturate the task; *no head can add information* |
| **B. End-to-end from scratch** (QCT) | **+0.007 AUC**, axion recall +0.06 | circuit participates in shaping the representation |
| **C. MAE pretrain → finetune** | **+0.010 AUC** (recipe-mismatched run) | same direction; absolute scores depressed by a training-recipe artifact |
| **QAE anomaly detection** on encoder features | quantum **0.9965 @ 72 params** vs sham 0.9956 @ 2,308 params | strongest quantum result: 32× parameter efficiency |

The mechanism is corroborated by the MAE paper itself: frozen MAE features are *not linearly
separable* (AUC 0.5365) and only become discriminative through fine-tuning — representation
shaping is where the task is actually solved, so a head-only comparison on finished features
cannot detect a circuit contribution even in principle.

**Caveats we state up front:** the regime-B/C gaps are single-seed and small; the 0.9965 QAE
result used features from a label-fine-tuned encoder (**label leakage** — invalid for an
unsupervised claim); and no parameter-matched (~72-param) classical control was ever run.
The proposed experiment fixes all three.

## 3. Proposed method: class-conditional QAE ensemble

```
64×64 lensing image
  → frozen MAE ViT encoder          (self-supervised, pretrained on no_sub only — zero labels)
  → 192-d CLS feature  (pad → 256)
  → amplitude embedding, 8 qubits
  → U(θ) → SWAP trash reset → U†(θ) → reconstruction fidelity      [72 params per model]

Train ONE such QAE per class (no_sub / cdm / axion), each on its own class only.
z-score each model's anomaly score on a held-out same-class calibration split:

  z_c(x) = (score_c(x) − μ_c) / σ_c
```

Three readouts from the same three trained models:

| Readout | Rule | Physics meaning |
|---|---|---|
| **Substructure anomaly** | z_no_sub as score | label-free detection that *any* substructure is present |
| **Generative 3-class** | argmin_c z_c | classification without a discriminative boundary (~216 params total) |
| **Open-set discovery** | min_c z_c large ⇒ unknown | flags a DM model absent from training — evaluated by leave-one-class-out rotation |

The open-set readout is the headline: a discriminative classifier is forced to pick one of
three classes and will confidently mislabel a fourth DM model; the generative ensemble can say
"none of the above."

## 4. Experimental design

**Arms (controlled comparison):**

| Arm | Description | Params/class | Answers |
|---|---|---|---|
| `quantum` | 8-qubit trash-qubit QAE | 72 | quantum performance |
| `sham` | classical Linear(256→K→256) AE, same fidelity metric | 2,308 | dimension-matched control |
| `matched` | fixed random orthogonal projection 256→8 + Linear(8→K→8) | **76** | **parameter-matched control — the honest efficiency test** |
| `maha` | per-class Mahalanobis (Ledoit-Wolf) on raw 192-d features | 0 (closed form) | are learned models needed at all? |

**Controls already built into the pipeline:**
- **Leakage-free features**: extracted with the self-supervised MAE encoder (`enc_I.pth`,
  pretrained on no_sub reconstruction only) — not the label-fine-tuned classifier.
- Identical data splits, calibration protocol, and model-selection rule
  (best epoch by calibration-split reconstruction, label-free) across all arms.
- The previously reported leaked-feature configuration is re-run as an explicit
  "leakage quantification" row.

**Pre-registered success criteria** (written before the full runs):

| Hypothesis | Pass criterion | If it fails |
|---|---|---|
| H1: leakage-free QAE still beats the AAE 0.93 literature line | anomaly AUC ≥ 0.95 | report as leakage-warning methodology result |
| H2: parameter efficiency survives the matched control | quantum ≥ matched at equal params | efficiency claim withdrawn |
| H3: open-set discovery works | open-set AUROC > 0.90 for held-out axion and cdm | report per-class asymmetry honestly |

## 5. Relation to prior DeepLense work

- Extends Alexander 2021 (AAE anomaly, 0.93) from binary anomaly to a 3-model generative
  ensemble with open-set capability, and replaces the classical AE with a 72-parameter QAE.
- Complements (does not compete with) the discriminative SOTA: closed-set accuracy is expected
  to stay below 0.968 — the contributions are the open-set capability, parameter efficiency,
  and the sham-control evaluation methodology itself.
- First systematic QML benchmark in the DeepLense ecosystem, with every number backed by a
  capacity-matched control.

## 6. Work plan

| Phase | Content | Status |
|---|---|---|
| 1 | Leakage-free feature extraction (`extract_features_ssl.py`) | ✅ done |
| 2 | Ensemble pipeline, 4 arms (`train_qae_ensemble.py`) | ✅ done, smoke-tested |
| 3 | Full single-seed run on Model_I | ✅ done — **H1/H2/H3 all failed**; see [docs/RESULTS.md §4b](docs/RESULTS.md) |
| 4 | Per pre-registered fallback: pivot the paper to the **leakage-quantification + sham-control methodology** angle; quantum-enhancement claims continue on the end-to-end line (regimes B/C) | active |
| 5 | Second dataset (Model_IV) — requires fixing a data-caching bug (all models at chance) | pending |
| 6 | Write-up: NeurIPS ML4PS 2026 workshop → journal extension | pending |

> **Outcome note (2026-06-10).** The decisive run quantified the leakage: the previously
> reported QAE anomaly AUC 0.9965 collapses to 0.438 on leakage-free self-supervised
> features, while a zero-parameter Mahalanobis baseline on the same features reaches 0.859.
> The pre-registered fallback applies: this proposal's empirical core becomes the
> leakage warning and the controlled-evaluation methodology; the open-set ensemble design
> remains valid but requires a score that does not saturate (Mahalanobis-class or flow-based)
> rather than overlap fidelity.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Leakage-free anomaly AUC drops well below 0.95 | the leakage exposure is itself a publishable methodology finding; AAE 0.93 remains the bar |
| Quantum ≤ matched classical at equal params | honest negative result within the sham-control framework — still the first controlled QML benchmark on DeepLense |
| Single-seed noise | effect sizes reported with the explicit single-seed caveat; multi-seed deferred by design |

## References

1. Prasha et al., *Masked Autoencoder Pretraining on Strong-Lensing Images...*, arXiv:2512.06642 (2025) — classical SOTA & upstream code
2. Alexander et al., *Decoding Dark Matter Substructure without Supervision*, arXiv:2008.12731 (2021)
3. Romero, Olson, Aspuru-Guzik, *Quantum autoencoders for efficient compression of quantum data*, QST 2, 045001 (2017)
4. Andrews et al., *Quantum Masked Autoencoders for Vision Learning*, arXiv:2511.17372 (2025)
5. Wang et al., *QVF: Neural Amplitude Encoding*, arXiv:2508.10900 (NeurIPS 2025)
6. Tesi et al. (incl. Gleyzer), *Quantum Attention for Vision Transformers in HEP*, arXiv:2411.13520 (2024)
7. Ngairangbam et al., *Anomaly detection in HEP using a quantum autoencoder*, PRD 105, 095004 (2022)
