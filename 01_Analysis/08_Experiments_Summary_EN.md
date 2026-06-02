# 08 — Experiments Summary: Methods Tried & Results (English)

> **Author:** [@leo07010](https://github.com/leo07010)
> **Compiled:** 2026-06-02
> **Purpose:** A consolidated, honest account of every method implemented and tested in this project (DeepLense × QML), with results and evidence strength.
> **Related:** method design in [`04_GSoC_QML_Proposal.md`](04_GSoC_QML_Proposal.md) and [`07_QMAE_DeepLense_Design.md`](07_QMAE_DeepLense_Design.md); upstream repo dissection in [`../02_Code/mae-lensing/REPO_ANALYSIS.md`](../02_Code/mae-lensing/REPO_ANALYSIS.md).
> **Note:** This is the English sibling of the Traditional-Chinese [`08_Experiments_Summary.md`](08_Experiments_Summary.md).

---

## TL;DR

Starting from the **2025 DeepLense MAE SOTA** (classical ViT, AUC 0.968), we systematically tested whether **attaching a quantum module helps or beats it**. The defining methodological choice: **every quantum method is paired with a capacity-matched "sham" (fake-quantum) control**. The honest finding: **on the discriminative task, with a strong classical baseline and abundant data, quantum yields no measurable gain (quantum ≈ sham)**. This motivated a **pivot from discrimination to generative / self-supervised reconstruction (QMAE)** as the genuine novelty.

**Evidence legend:** ✅ measured numbers · 🟡 qualitative conclusion / docstring reference value · ⏳ code complete but results not yet recorded

---

## Methodological backbone (applies to every quantum experiment)

Two design principles make the comparisons fair and the conclusions trustworthy:

1. **Sham control.** For each quantum head, a `--sham` variant replaces the quantum circuit with a classical projection of **matched dimensionality / token count** — same architecture, no circuit. *Quantum vs. sham under an identical wrapper is the only clean test of quantum advantage*: if sham ≈ quantum, any gain came from the surrounding (classical) machinery, not the qubits.
2. **Gated residual init.** Fusion heads add the quantum branch onto a frozen classical-baseline head through a `tanh(gate)` term with `gate = 0` at init ⇒ **the model starts exactly at the classical baseline (AUC ≈ 0.974) and can only improve.** This removes "the quantum head just trained worse" as a confound.

All quantum experiments run on the **same frozen MAE ViT encoder features** (192-d CLS, 256 patch tokens, or 16×16 downsampled images) cached once, so every method is compared on identical inputs.

---

## Stage 0 — Classical baseline & analysis

| # | Method | File | Goal | Result | Evidence |
|---|---|---|---|---|---|
| 0a | Evaluate MAE pretrained classifier | `eval_pretrained.py` | Measure the target to beat | **AUC 0.974 / acc 88.65%** | ✅ |
| 0b | Per-sample error analysis (8,910 val) | `analyze_errors.py` + `error_analysis/` | Locate the classical model's weakness | **acc 0.8847; axion recall 0.769, cdm 0.955, no_sub 0.928; dominant confusion axion→cdm (677 images)** | ✅ |
| 0c | Feature extraction | `extract_features.py`, `extract_patch_features.py`, `extract_images16.py` | Cache 192-d CLS / 256 patch tokens / 16×16 images for the quantum experiments | Reusable caches produced | ✅ |
| 0d | t-SNE visualization | `enhanced_tsne.py` | Inspect representation separability | axion/cdm clusters overlap (consistent with 0b) | ✅ |

**Why this matters:** Stage 0 fixes both the target (AUC 0.974) and the classical model's known weakness (axion recall ≈ 0.77, driven by axion↔cdm confusion). Every later quantum experiment reuses these frozen features for an apples-to-apples comparison.

---

## Stage 1 — Quantum **discriminative** experiments (on the frozen MAE encoder)

All three use the gated-residual init (start = baseline AUC 0.974) and ship a `--sham` control.

| # | Method | File(s) | Quantum design | Result | Evidence |
|---|---|---|---|---|---|
| 1 | **F2 — Gated Residual Fusion** | `quantum_fusion_cudaq.py`, `quantum_fusion_pennylane.py` | 16-qubit dressed PQC (RY encode once → [RZ, RY + CNOT-ring] × 4 → ⟨Z⟩), gated-residual onto a classical linear head | Folded into the "quantum ≈ sham" conclusion | 🟡 |
| 2 | **Cross-Attention Mid-Fusion** | `quantum_fusion_xattn.py` | Re-implements Alavi et al. (arXiv:2512.19180): 32 quantum readout tokens + CLS token through self-attention (variants `reupload` / `pure` / `sham`) | Folded into the "quantum ≈ sham" conclusion | 🟡 |
| 3 | **QCT — Quantum-Classical Transformer** | `quantum_fusion_qct.py` + `train_qct.py` | Token-level full fusion: 256 ViT patch tokens + 32 quantum tokens + CLS through one mixed self-attention stack | **Explicitly recorded: QCT quantum = QCT sham** | 🟡 |

### Stage 1 headline finding (from [`07_QMAE_DeepLense_Design.md`](07_QMAE_DeepLense_Design.md))

> Across the gated / xattn / QCT experiments (each with a sham control): **with a strong classical baseline and abundant data, quantum produces no measurable gain on the discriminative task (QCT quantum = QCT sham).** Any improvement came from the fusion architecture itself, not the quantum circuit.

This is the negative result the proposal pre-framed as *"characterize where it helps, don't assume it wins"* — itself a valuable community contribution.

### Side result — gradient-engine benchmark (from docstrings)

Measured on H100, 16 qubits, batch 64:

| Engine | Speed |
|---|---|
| CUDA-Q parameter-shift | 57 s/batch |
| lightning.gpu + adjoint | 18 s/batch |
| **`default.qubit` + backprop (fully batched torch on GPU)** | **0.33 s/batch (~170× faster)** |

→ All later experiments switched to `default.qubit + backprop`.
Reference value noted in code: axion recall — baseline 0.754 → classical-only retrained 0.788 (the quantum line did not surpass this).

---

## Stage 2 — Quantum **generative / self-supervised** experiments (the post-pivot novelty)

| # | Method | File(s) | Design | Benchmark target | Result | Evidence |
|---|---|---|---|---|---|---|
| 4 | **QMAE (Quantum Masked Autoencoder)** | `quantum_mae.py`, `train_qmae.py`, `train_qmae_cls.py` | Faithful to Andrews et al. (arXiv:2511.17372): 16×16 → amplitude-embed 8 qubits → U(θ) → SWAP latent (trash naturally reset) → U†(θ) → reconstruction fidelity (self-supervised, label-free); downstream latent ⟨Z⟩ → 3-class + sham | quantum baseline + sham (**not** 0.968) | not yet recorded | ⏳ |
| 5 | **Quantum AE anomaly detection** | `train_qae_anomaly.py`, `train_qae_anomaly_cls.py` | Romero 2017 trash-qubit compression; train on no_sub only, reconstruction fidelity = normality score; two inputs: 16×16 pixels and 192-d CLS features | Alexander 2021 AAE ≈ 0.93 | not yet recorded | ⏳ |
| 6 | **Equivariant quantum residual + few-shot sweep** | `quantum_equiv.py`, `train_fewshot.py` | C4 group-averaged → rotation-invariant quantum features; N = 25/50/100/250/500 per class comparing classical / sham / quantum | sham (small-N regime) | not yet recorded | ⏳ |

**Positioning (doc 07):** proof-of-concept, benchmarked against a "quantum baseline + capacity-matched classical sham" — **not** an attempt to beat the classical MAE's 0.968 (the QMAE paper reaches only ~65% even on MNIST). The contribution is: **first QMAE applied to strong-lensing dark-matter data + rigorous sham controls + few-shot scaling analysis**, extending arXiv:2511.17372 from MNIST to real scientific data.

---

## One-page overview

| Stage | # methods | Core finding | Evidence strength |
|---|---|---|---|
| 0 — classical baseline + analysis | 4 | AUC 0.974; bottleneck is axion recall 0.77 (axion↔cdm confusion) | ✅ measured |
| 1 — quantum discriminative (3 architectures × {quantum, sham, variants}) | 3 | **quantum ≈ sham, no measurable quantum gain** | 🟡 qualitative |
| 2 — quantum generative / SSL | 3 | code complete, framed as proof-of-concept | ⏳ not yet run to completion |

---

## Result-persistence status (honest disclaimer)

- **Only the classical baseline numbers are actually stored**: `02_Code/mae-lensing/error_analysis/` (`error_analysis.csv` + figures).
- **The quantum lines (Stages 1 & 2) have no per-run numbers committed**: no `outputs_*/` directories, no training logs, no result CSVs.
  - Stage 1 rests on the prose conclusion in doc 07 plus the docstring reference values in each `quantum_fusion_*.py`.
  - Stage 2's three lines are code-complete but still marked ⏳ (to be run) in the README milestones.

---

## Recommended next steps

1. **Back-fill Stage 1 with real numbers** — produce a `classical / sham / quantum × {gated, xattn, QCT}` AUC table; this is the most convincing evidence for the negative result.
2. **Run Stage 2's three generative lines to completion**, especially the `train_fewshot.py` small-N curve (the regime where quantum is most likely to beat sham).
3. **Add a results logger** — dump each training run's stdout to structured CSV/JSON (matching the existing `error_analysis/` format), so conclusions are never again recorded without their numbers.
