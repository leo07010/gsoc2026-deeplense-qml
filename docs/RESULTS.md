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

## 5. Model_IV dataset — pipeline bug (open)

Every model (classical / quantum / sham, all architectures) scores AUC ≈ 0.50
on the cached Model_IV data, while the same code reaches 0.95–0.98 on Model_I.
MAE pretrain loss also plateaus immediately (0.004 vs 0.001 on Model_I).
Conclusion: the `cache_model.py` preprocessing (per-image min-max norm) or the
Model_IV npy structure breaks the signal — **data bug, not a model result**.

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
