# Consolidated Experimental Results

> All numbers below are extracted from SLURM logs of completed runs
> (HPC, H100/H200, PennyLane `default.qubit` + backprop).
> Single seed (42) unless noted. Dataset: DeepLense Model_I-style 3-class
> (axion / cdm / no_sub) unless noted.
> **Sham** = identical architecture with the quantum circuit replaced by a
> classical projection of matched dimensionality.

## 0. Classical reference points

| Item | Value | Source |
|---|---|---|
| MAE paper SOTA (mask 0.9, finetuned) | AUC 0.9681 / acc 88.65% | arXiv:2512.06642 Table 3 |
| MAE paper, **frozen** encoder + linear head | **AUC 0.5365** / acc 34.06% | arXiv:2512.06642 Table 1 — "frozen features are not linearly separable; fine-tuning is essential" |
| Shipped fine-tuned checkpoint, linear probe | AUC 0.9734 / acc 88.11% | our re-evaluation |
| Classical head retrained on frozen features | AUC 0.9797 | `classical_control.py` |
| Classical error profile | axion recall 0.769, axion→cdm = dominant confusion | `results/error_analysis/` |
| Alexander 2021 AAE anomaly (literature) | AUC ≈ 0.93 | arXiv:2008.12731 |

## 1. Regime A — quantum heads on FROZEN fine-tuned features

All heads sit on 192-d CLS features from the label-fine-tuned classifier
(linear probe already 0.9734 → ceiling saturated).

| Architecture | Quantum AUC | Sham AUC | Δ |
|---|---|---|---|
| Gated residual fusion | 0.9797 | ≈ 0.9797 | 0 |
| Cross-attention fusion | 0.9820 | 0.9822 | −0.0002 |
| QCT (256 patch + 32 quantum tokens) | 0.9838 | 0.9838 | 0 |
| QVF (learnable neural amplitude encoding) | 0.9816 | 0.9822 | −0.0006 |

**Conclusion: exact tie everywhere.** With saturated features, no head — quantum
or classical — can add information. This regime cannot detect a circuit
contribution even in principle.

## 2. Regime B — END-TO-END from scratch (raw 64×64 images)

| Architecture | Quantum | Sham | Δ AUC | Axion recall (Q vs S) |
|---|---|---|---|---|
| QCT-scratch | **0.9605** | 0.9533 | **+0.0072** | **~0.90 vs ~0.84** |
| QVF-scratch | 0.9804 | 0.9796 | +0.0008 | ≈ |

The QCT gap concentrates on axion — the known classical bottleneck class.
Single seed; treat as directional evidence, not a confirmed effect.

## 3. Regime C — MAE pretrain → finetune (recipe-mismatched run)

MAE pretrain on no_sub (mask 0.9, 15 ep) → finetune all classes, 20 ep.
⚠️ Finetune used AdamW lr 5e-4 / wd 1e-4 / batch 128 / label smoothing —
the paper recipe is **Adam lr 5e-5 / wd 1e-5 / dropout 0.1 / batch 64 / 10 ep**.
Quantum/sham heads also pass a 192→8 bottleneck the classical head does not have.

| Head | Best AUC | Note |
|---|---|---|
| Classical (192→3 direct) | 0.9799 | reproduces paper SOTA |
| Quantum (192→8→PQC→3) | 0.9012 | **quantum > sham by +0.0098** |
| Sham (192→8→tanh→3) | 0.8914 | both depressed by bottleneck + recipe |

First ~6 epochs of the quantum run sit at chance (circuit hard to train at 10×
the paper's learning rate). A recipe-aligned rerun is the designed fix.

## 4. Generative / self-supervised lines

| Experiment | Quantum | Sham | Note |
|---|---|---|---|
| QMAE downstream 3-class (16×16) | 0.9581 | 0.9743 | sham wins |
| QAE anomaly, raw 16×16 pixels | 0.5203 | 0.5516 | both fail — downsampling destroys the signal |
| **QAE anomaly, 192-d CLS features** | **0.9965 @ 72 params** | 0.9956 @ 2,308 params | ⚠️ features label-leaked (fine-tuned encoder) — re-run on SSL features in progress |
| Few-shot (frozen feats), N=500/class | 0.8389 | 0.9077 | sham wins; frozen-regime caveat applies |

## 4b. QAE ensemble on LEAKAGE-FREE SSL features (2026-06-10, the decisive test)

Class-conditional one-class AE per class on features from the **self-supervised**
MAE encoder (`enc_I.pth`, no labels anywhere). Full pipeline:
`experiments/04_qae_ensemble/`. Raw numbers: `results/qae_ensemble_results.csv`.

| Arm | Params/class | 3-class AUC | Anomaly AUC | Open-set mean |
|---|---|---|---|---|
| **Mahalanobis** (closed-form) | 0 | **0.913** | **0.859** | 0.563 |
| Sham AE (dim-matched) | 2,308 | 0.593 | 0.571 | 0.520 |
| Matched AE (param-matched) | 76 | 0.533 | 0.496 | 0.512 |
| Quantum QAE | 72 | 0.470 | 0.438 | 0.474 |

**Pre-registered hypotheses — all three failed:**

- **H1 failed:** leakage-free anomaly AUC is 0.44–0.57 for every learned AE — the previously
  reported **0.9965 was label leakage**, quantified at ≈ 0.44 AUC of inflation.
- **H2 failed:** quantum (0.438) < param-matched classical (0.496) at equal parameters.
- **H3 failed:** open-set AUROC ≈ 0.5 for all learned arms.

**Mechanism:** the overlap-fidelity score saturates — classical AEs reconstruct *every*
sample at fidelity ≈ 0.9999 (calibration σ ≈ 1e-4), the quantum AE compresses *nothing*
(fidelity ≈ 0.34, σ ≈ 2e-3); neither produces a discriminative score. Meanwhile Mahalanobis
on the raw 192-d features reaches 0.859/0.913 — **the SSL features do contain the signal;
the 8-dim fidelity-AE design cannot extract it.**

**Standing conclusions:** (i) frozen-feature anomaly results computed on label-fine-tuned
encoders are untrustworthy — a quantitative leakage warning for the field; (ii) the
QAE-on-features route is dead in its current form; the live quantum-vs-sham direction
remains the end-to-end regime (Sections 2–3).

## 5. Model_IV dataset — RESOLVED: not a bug, a genuinely hard open problem

Every model (classical / quantum / sham, all architectures) scores AUC ≈ 0.50 on
the cached Model_IV data. Investigation (2026-06-10) ruled out every artifact:

| Hypothesis | Test | Verdict |
|---|---|---|
| Cache/label corruption | MD5 duplicate check + visual inspection | ✗ clean — no duplicates, images healthy |
| Training recipe mismatch | exact paper recipe (`pretrain_finetune_v2.py`): Adam 5e-5, wd 1e-5, dropout 0.1, batch 64, 10 ep | ✗ classical still 0.5030 (same recipe reproduces 0.9626 ≈ paper's 0.968 on Model_I) |
| "MAE paper did 0.97 on this data" | downloaded the paper's actual Dataset1 and inspected | ✗ **Dataset1 is Sérsic-source data (Model_II/III family) — full bright Einstein rings. It is NOT Model_IV.** |

Model_IV (real-galaxy sources + Euclid systematics; thin partial arcs, point-like
images, extreme morphological diversity) has **no published 3-class result ≥ 0.9
anywhere in the DeepLense ecosystem** — the GSoC "SSL from real dataset" project
(iBOT AUC 0.99) is a *different task* (binary lens-finding on 3-channel survey
data). Model_IV 3-class substructure is an open problem, consistent with
Tsang 2024 (realistic sources are dramatically harder). Standing AUC: ~0.50.

## 6. Gradient-engine benchmark (16 qubits, batch 64, H100)

| Engine | s/batch |
|---|---|
| CUDA-Q parameter-shift | 57 |
| lightning.gpu + adjoint | 18 |
| **default.qubit + backprop (batched torch)** | **0.33** |

## Honest-evidence legend

- Regime A ties: multiple architectures, consistent — robust.
- Regime B/C quantum>sham gaps: **single seed, small** — directional only.
- QAE 0.9965: leaked features — the leakage-free re-run
  (`experiments/04_qae_ensemble/`) is the current experiment.
