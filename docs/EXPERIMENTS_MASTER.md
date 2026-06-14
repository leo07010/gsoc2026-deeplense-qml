# Master Experiment Log — Quantum vs Classical on DeepLense

> Every quantum method is paired with a **capacity-matched classical "sham"**
> (same architecture, circuit → classical layer of matched dimensionality).
> `quantum − sham` is the only clean test of a genuine circuit contribution.
> Classical SOTA reference: MAE-pretrained ViT, AUC 0.968 (arXiv:2512.06642).
> All numbers from completed SLURM runs (H200, PennyLane default.qubit+backprop).

## 0. The methodology

Two principles make every comparison honest:
1. **Sham control** — isolates the circuit from the classical wrapper.
2. **Geometric-difference certificate** (Huang et al. 2021) — `g(K_C,K_Q)`: a
   *certificate* that, when `g ≪ √N`, **no** quantum-kernel advantage is possible
   for **any** labels — not "we didn't find one", but "it cannot exist".

## 1. Classical baselines (the target)

| Dataset | MAE pretrain→finetune | ViT scratch |
|---|---|---|
| Dataset1 (paper's data) | **0.9672** (reproduces paper 0.968) | 0.9657 |
| Model_I | 0.9633 | 0.9243 |
| Model_II | 0.9682 | 0.9660 |

## 2. Quantum discriminative — every architecture × sham

| Method | Where quantum sits | Quantum | Sham | Δ | Verdict |
|---|---|---|---|---|---|
| Gated / X-attn / QCT (frozen feats) | readout head | 0.982–0.984 | 0.982–0.984 | ≈0 | **tie** (feature ceiling saturated) |
| pretrain→finetune NAE head | readout head | 0.503 | 0.496 | — | NAE saturates → dead |
| QFair qct (wd/init/lr fixed, qlr swept) | end-to-end head | 0.9790 | 0.9802 | −0.0012 | **tie** |
| QFair qvf (pathology-fixed) | end-to-end head | 0.9838 | 0.9830 | +0.0008 | **tie** |
| QCT-scratch (I / II / D1) | end-to-end token | .951/.976/.975 | .952/.970/.964 | ±noise | **tie** |
| QVF-scratch (I / II / D1) | end-to-end encoder | .981/.998/.998 | .979/.993/.996 | +small | **tie** |
| QViT (quantum inside ViT encoder) | mid-encoder | 0.962 | 0.970 | −0.008 | **quantum loses** (surrogate theorem) |
| Dual-encoder FiLM (qct / qvf) | two-tower fusion | 0.979/0.984 | 0.981/0.984 | ≈0 | **tie** |

**The earlier +0.0072 (QCT, single seed) did NOT replicate** — on re-run it became
−0.0012. Isolated positives flip sign across seeds/datasets ⇒ noise.

## 3. Quantum kernels — qubit scaling (8→12→16), GPU exact statevectors

| Kernel | g_min vs best classical | advantage threshold 2√N ≈ 57 |
|---|---|---|
| Fidelity (n=8/12/16) | 5.7 → 2.7 → **2.3** (shrinks) | far below — and getting worse (exponential concentration) |
| Projected PQK (n=8/12/16) | 11.5 → 12.3 → **10.1** | far below |

Few-shot SVM (convex, no training issue): quantum kernel loses to RBF at every
N and every qubit count. **Certificate result: kernel advantage is impossible on
this data, and adding qubits does not help.**

## 4. Generative (IQP Born machine vs param-matched classical), exact MLE

| n_bits | IQP (quantum) | Ising (matched) | AR | MoB | cat |
|---|---|---|---|---|---|
| 10 | 6.955 | 6.869 | 6.868 | **6.710** | 6.432 |
| 12 | 8.449 | 8.237 | 8.237 | **8.000** | 7.689 |

(held-out NLL, lower=better). **IQP is the worst of the matched family** —
interference hurts on low-order natural latents.

## 5. Anomaly detection (leakage-free SSL features)

| Arm | params/class | anomaly AUC |
|---|---|---|
| Mahalanobis (0-param) | 0 | **0.859** |
| Sham AE | 2,308 | 0.571 |
| Matched AE | 76 | 0.496 |
| Quantum QAE | 72 | **0.438** |

The previously reported **0.9965 was label leakage** (fine-tuned encoder); the
clean number is 0.438 — quantum is the worst learned arm.

## 6. Robustness battery (clean-train, perturbed-eval) — no consistent edge
## 7. Few-shot end-to-end (2 arch × 2 data × 5 seeds) — no consistent edge
## 8. Multi-view fusion M=1→8 — Δ(q−sham) stays ≈0/negative, no positive slope

## 9. Equivariant 2×2 (REQAE) — the one place something moved

| N/class | q-equiv | c-equiv | q-plain | c-plain | Δ(equiv−plain) |
|---|---|---|---|---|---|
| 50 | 0.652 | 0.656 | 0.525 | 0.554 | **+0.114** |
| 100 | 0.781 | 0.782 | 0.583 | 0.619 | **+0.181** |

**C4 rotation-invariance is a large low-data win (+0.11–0.18)** — but
`q-equiv ≈ c-equiv`: the win is the **symmetry, not the circuit** (the Chang-2023
lesson). Quantum adds nothing over a classical-equivariant layer of matched size.

## 10. Training diagnostic — is the circuit actually trained? (instrumented)

| metric | result | meaning |
|---|---|---|
| circuit grad norm | 0.12–0.32 (non-zero) | no barren plateau |
| weight drift ‖w−w₀‖ | grows 0.08→8.5 | weights move a lot |
| output std (⟨Z⟩) | 0.05–0.09 | outputs vary (informative) |
| CNN grad norm | 2–5 | gradient flows upstream |
| **AUC with circuit zeroed** | **0.5000 (chance)** | **the circuit is the entire classifier** |

**The "you didn't train the quantum" hypothesis is refuted by direct
measurement.** The circuit is trained, used, and is the sole decision pathway —
yet it still ties the sham, because the sham computes the same function.

---

## The map in one line

| Battlefield | Result |
|---|---|
| Discriminative (12 architectures × sham) | tie |
| Pathology-fixed training (wd/init/lr/readout) | tie |
| Fidelity kernel, 8/12/16 qubits | certificate: impossible |
| Projected kernel, 8/12/16 qubits | certificate: impossible |
| Few-shot (convex SVM + end-to-end) | no edge |
| Robustness | no edge |
| Generative IQP | loses to matched Ising |
| Multi-view fusion M=1→8 | no positive slope |
| Quantum-in-encoder (QViT) | quantum loses |
| Equivariant 2×2 | symmetry wins, **not** the circuit |
| Training audit | circuit IS trained correctly |

**Conclusion.** On classical-simulator-generated strong-lensing images, at
simulator-accessible scale (≤16 qubits), with capacity-matched controls and a
verified-correctly-trained circuit, **no quantum advantage exists** — and the
geometric-difference certificate explains *why* (g ≪ √N: classical kernels span
the relevant function space). This reproduces the conclusion of the largest
rigorous QML benchmarks (Bowles 2024; Schnabel 2025, 20 000 models) with an
added certificate and a cross-8-battlefield map.
