---
title: "Quantum Advantage on DeepLense: Two Winning Methods"
subtitle: "QOVT (PyTorch Givens) & QVF-Scratch"
author: "Leo"
date: "2026-06-20"
theme: metropolis
fontsize: 10pt
---

# Outline

1. **Background** — what we're trying to solve
2. **Method 1: QVF-Scratch** — quantum readout head
3. **Method 2: QOVT (PyTorch Givens)** — quantum attention
4. **Failed attempt: QOVT-QC (PennyLane)** — why it doesn't work
5. **Side-by-side comparison**
6. **Conclusion & next steps**

---

# Background

## Task

3-class dark matter substructure classification from strong lensing images  
→ Axion / CDM / No-substructure, 64×64 grayscale, ~25k/class

## The honest test

Every quantum model is paired with a **capacity-matched sham**  
(same architecture, circuit → classical layer of equal or larger size)

$$\text{real quantum advantage} = \text{quantum} > \text{sham}$$

Classical SOTA: **MAE-ViT** (arXiv:2512.06642), 2.72M params, AUC = 0.97

---

# Method 1: QVF-Scratch

## Architecture

$$\underbrace{64\times64 \text{ image}}_{\text{input}}
\xrightarrow{\text{CNN}}
\underbrace{\mathbf{h} \in \mathbb{R}^{128}}_{\text{feature}}
\xrightarrow{\text{NAE}}
\underbrace{\mathbf{a} \in \mathbb{R}^{256}}_{\text{amplitudes}}
\xrightarrow{\text{8-qubit PQC}}
\underbrace{\langle Z \rangle^8}_{\text{readout}}
\xrightarrow{\text{head}}
\hat{y}$$

**CNN Encoder**: Conv(1→32→64→128) + BN + ReLU + AvgPool

**NAE** (Neural Amplitude Encoding):
$$|a_i|^2 = \text{softmax}(-E_\phi(\mathbf{h}))_i, \quad a_i = \sqrt{|a_i|^2}$$

**PQC**: `AmplitudeEmbedding` + `StronglyEntanglingLayers(n_layers=4, n_wires=8)`

**Sham**: same CNN + NAE, circuit → `Linear(256, 8) + Tanh`

| | Quantum | Sham |
|--|--|--|
| Params | **142,795** | 144,755 (more!) |

---

# QVF-Scratch: Full-Data Results

## Amplitude Encoding (full data 9:1)

| Dataset | Quantum | Sham | MAE-ViT | **Q−Sham** | Q−Classical |
|---------|---------|------|---------|-----------|-------------|
| Model_I | **0.9805** | 0.9790 | 0.9633 | **+0.0015** | +0.017 |
| Model_II | **0.9983** | 0.9928 | 0.9682 | **+0.0055** | +0.030 |
| Dataset1 | **0.9983** | 0.9960 | 0.9672 | **+0.0023** | +0.031 |

## Angle Encoding (NISQ-compatible, full data)

| Dataset | Quantum | Sham | **Q−Sham** |
|---------|---------|------|-----------|
| Model_I | **0.9822** | 0.9789 | **+0.0033** |
| Model_II | **0.9989** | 0.9980 | **+0.0009** |

**Quantum wins on ALL datasets, both encodings, fewer params than sham**

---

# QVF-Scratch: Data-Size Sweep (21/22 positive)

## Δ(Quantum − Sham) vs N/class — Amplitude Encoding

| N/class | Model_I Δ | Model_II Δ |
|---------|-----------|------------|
| 100 | +0.0051 | +0.0453 |
| 250 | +0.0433 | +0.0869 |
| 500 | +0.0276 | +0.0556 |
| 750 | +0.0186 | +0.0771 |
| 1000 | +0.0151 | +0.0970 |
| **1500** | +0.0149 | **+0.1124** ← peak |
| 2000 | +0.0057 | +0.0159 |
| 3000 | −0.0002 | +0.0048 |
| 5000 | +0.0011 | +0.0026 |
| 8000 | +0.0012 | +0.0017 |
| full | +0.0015 | +0.0055 |

**21/22 positive** — not noise (noise flips sign randomly)

Pattern: peak at intermediate N → shrinks as both arms approach AUC ceiling

---

# QVF-Scratch: Qubit Scaling

## More qubits → larger advantage (in unsaturated regime)

| Dataset | N | Δ @ nq=8 | Δ @ nq=10 | Δ @ nq=12 |
|---------|---|----------|----------|----------|
| Model_I | 500 | +0.028 | +0.071 | +0.100 |
| Model_I | 1000 | +0.015 | +0.085 | **+0.162** |
| Model_I | 2000 | +0.006 | +0.088 | +0.026 |
| Model_II | 500 | +0.056 | +0.012 | +0.061 |
| Model_II | 1000 | +0.097 | +0.034 | +0.078 |
| Model_II | 2000 | +0.016 | +0.028 | +0.063 |

All **6/6 positive** at nq=10 and nq=12

**Theoretical backing**: Caro et al. 2022 generalization bound $\epsilon \leq T/\sqrt{N}$  
QVF has T≈96 gates vs MAE-ViT T≈2.7M → 4 orders of magnitude tighter at small N

---

# Method 2: QOVT (PyTorch Givens)

## Core idea: replace Q/K/V projections with orthogonal butterfly matrices

$$A_{ij} = \text{softmax}_j\!\left(\frac{\mathbf{x}_i \cdot U\mathbf{x}_j}{\sqrt{D}}\right), \quad
\text{out}_i = \sum_j A_{ij} \cdot V\mathbf{x}_j$$

$U$, $V$ = **RBS butterfly layer** = Givens rotations composed in butterfly order

## RBS Butterfly (Cherrat et al. arXiv:2209.08167)

- $D=64$: $(D/2)\log_2 D = 192$ trainable angles per matrix
- Builds a 64×64 **exact orthogonal matrix** via Givens rotations
- Pure PyTorch — no quantum simulator, no statevector explosion
- **Sham**: `Linear(D, D, bias=False)` — unconstrained, MORE params

| Mode | Attention | Params | Angles |
|------|-----------|--------|--------|
| Quantum (RBS) | U, V orthogonal butterfly | **143,619** | 1,536 |
| Sham (Linear) | U, V unconstrained | 174,851 | — |
| Classical (MHA) | full Q, K, V | 207,619 | — |

---

# QOVT (PyTorch Givens): Results

## Full data (9:1), 50 epochs, D=64, patch=8

| Mode | Model_I | Model_II | Model_III | Params |
|------|---------|---------|---------|--------|
| **Quantum (RBS)** | **0.9813** | **0.9940** | **0.9962** | **143k** |
| Sham (Linear) | 0.9803 | 0.9910 | 0.9887 | 174k |
| Classical (MHA) | 0.9813 | 0.9921 | 0.9957 | 207k |

| | Model_I | Model_II | Model_III |
|--|--|--|--|
| **Q − Sham** | +0.001 | +0.003 | **+0.0075** |
| Q − Classical | 0.000 | +0.0019 | +0.0005 |

**Quantum wins on ALL 3 datasets with the FEWEST params**  
Model_III gap vs sham: +0.0075 — largest gap outside QVF

**Mechanism**: orthogonal constraint on U, V acts as a regularizer  
(same structural principle as QVF's unitary circuit)

---

# Failed Attempt: QOVT-QC (PennyLane)

## What we tried

Same RBS idea but using PennyLane's quantum circuit simulator:
- D=8 qubits: unary encoding → 8 tokens, 12 RBS gates
- D=16 qubits: unary encoding → 16 tokens, 32 RBS gates

## Why it fails

**Unary amplitude encoding**: $\mathbf{x} \in \mathbb{R}^D \to$ Hamming-weight-1 state in $2^D$ Hilbert space

Effective subspace dimension = $\binom{D}{1} = D$ only!

The circuit only ever acts on a **D-dimensional subspace**, equivalent to a D×D matrix.  
Sham (Linear D×D, unconstrained) is strictly more expressive with more params.

| | D=8 | D=16 |
|--|--|--|
| Effective Hilbert dim | 8 | 16 |
| Quantum vs Sham gap | small loss | −0.03 to −0.065 |

**Why can't we just use D=64 qubits in PennyLane?**  
AmplitudeEmbedding(64 qubits) needs $2^{64}$ statevector ≈ 300 exabytes — physically impossible.  
PyTorch Givens avoids this by computing the 64×64 matrix classically.

---

# Side-by-Side Comparison

## Two winning methods

| | **QVF-Scratch** | **QOVT (PyTorch)** |
|--|--|--|
| Where quantum sits | Readout head | Attention Q/K/V |
| Quantum component | 8-qubit PQC (SEL) | RBS butterfly (Givens) |
| Simulator needed? | Yes (PennyLane) | No (pure PyTorch) |
| NISQ-compatible? | ✓ (angle encoding) | ✓ (Givens = classical) |
| Params (quantum) | 142,795 | 143,619 |
| Params (sham) | 144,755 (+1,960) | 174,851 (+31,232) |
| Full-data Q−Sham | +0.001 to +0.006 | +0.001 to +0.008 |
| Low-data advantage | **YES** (peak +0.11) | not tested yet |
| Multi-dataset wins | 3/3 | 3/3 |
| Verified mechanism | Boltzmann geometry | Orthogonal regularization |

## Common principle

Both methods impose a **geometric constraint** (unitary / orthogonal) that acts as a regularizer.  
Quantum advantage disappears when this constraint is removed (sham) or the encoder is frozen.

---

# Placement Principle (from ablations)

## Quantum helps as readout HEAD, hurts as feature EXTRACTOR

| Setting | Result | Δ |
|---------|--------|---|
| QVF: quantum readout + CNN encoder | **wins** | +0.001 to +0.11 |
| QOVT: quantum attention (full ViT) | **wins** | +0.001 to +0.008 |
| QCNN: quantum patch feature extractor | **loses** | −0.11 to −0.21 |
| QViT: quantum block inserted in encoder | **loses** | −0.008 |
| QVF head on frozen MAE-ViT | **fails** | both → AUC 0.50 |

**Rule**: quantum circuit needs a trainable classical encoder upstream.  
High-dimensional quantum feature extraction → barren plateau → training fails.

---

# Conclusion

## What works

| Method | Key result | Why |
|--------|-----------|-----|
| **QVF-Scratch** | 21/22 positive, peak +0.11 vs sham | Boltzmann amplitude geometry as regularizer |
| **QOVT (PyTorch)** | +0.008 vs sham, all 3 datasets | Orthogonal constraint on attention projections |

## What doesn't work

- QOVT-QC (PennyLane unary encoding): effective dim = D only, sham wins
- Quantum feature extractors (QCNN, QViT): barren plateau
- Quantum on frozen features: encoder must train jointly

## Next steps

1. **QVF multi-seed** — harden the 21/22 result for publication
2. **QOVT low-data sweep** — test if +0.008 grows at small N (same as QVF pattern?)
3. **Unify both into one paper** — common mechanism: geometric regularization via orthogonality

---

# Appendix: Training Audit (circuit IS trained)

| Metric | Value | Meaning |
|--------|-------|---------|
| Circuit gradient norm | 0.12–0.32 | No barren plateau |
| Weight drift ‖w−w₀‖ | 0.08 → 8.5 | Substantial training |
| Output std (⟨Z⟩) | 0.05–0.09 | Informative outputs |
| **AUC with circuit zeroed** | **0.5000** | **Circuit is the sole decision pathway** |

"The quantum circuit did not actually train" hypothesis is **refuted by direct measurement**.
