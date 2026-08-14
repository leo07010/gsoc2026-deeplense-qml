# Pre-registration: Quantum Patch Embedding (QPE)

**Date committed: 2026-08-14, BEFORE any sweep run of this experiment.**
Smoke tests (2 epochs, 128 images, single seed) have run to verify the
code executes; no sweep data existed when this document was committed.

## Background and motivation

The Step-0 spectral diagnostic (`experiments/09_spectral_diagnostic/`,
commit 619fae7) showed:

1. The three classes (axion / cdm / no_sub) separate strongly in
   **intra-patch spatial frequency** (radial bins 7-29, wavelengths
   2-9 px; Cohen's d up to 2.86 on log-power, both Model_II and
   Model_III).
2. They do **not** separate along the 8x8 token-grid axis (max d < 0.4):
   mean-pooling to the token grid destroys the signal.

Consequence: the earlier idea of a quantum mixer on the token axis is
dead (refuted by its own pre-committed gate). Instead, the RY+CNOT
circuit is placed at the **patch-embedding stage**, acting on the raw
64-pixel vector of each 8x8 patch, where its bit-stride structure is a
learnable intra-patch multiscale 2D filter aligned with where the signal
actually lives (qubit q mixes pixels at spatial stride 2^(bit) within
the patch; see `train_qpe.py` docstring for the exact bit map).

This experiment asks ONE question the QOVT ablation could not:
**does the circuit's bit-space structure itself do real work, beyond
orthogonality and low parameter count?**

## Arms (all share a byte-identical classical ViT backbone, 203,459
backbone params; only the patch embedding differs)

| arm | embedding | embed params |
|---|---|---|
| quantum | RY+CNOT circuit (6 qubits, 8 layers) on raw pixel vector, bit-aligned | 49 |
| scramble | same circuit, pixel->basis assignment randomly permuted (per training seed) | 49 |
| dct | fixed orthonormal 2D DCT-II (classical spectral analog) | 1 |
| conv | learned Conv2d(1,64,8,8) (the standard ViT patch embed) | 4,160 |

Design constraint (deliberate, do not "fix"): no learned linear sits
between embedding and first attention block for quantum/scramble/dct --
it would absorb any orthogonal transform and erase the measured
difference. The three non-conv arms each carry ONE scalar learnable gain
(hence 49/49/1 above): adversarial review found conv's unconstrained
weight norm lets it tune the token-to-positional-embedding ratio while
norm-preserving embeds cannot, which would confound tests 3-4; a scalar
gain closes this and cannot absorb an orthogonal transform. Backbone
construction order also fixed so all 4 arms get bit-identical backbone
init per seed (embed built last; quantum/scramble were already exactly
paired, now dct/conv are too).

## Protocol

- Datasets: Model_II, Model_III. N=500/class. Seeds 42-61 (**20 seeds**;
  raised from the QOVT ablation's 10 after a power analysis on that
  ablation's own observed paired-difference SDs of 3-17pp showed n=10
  detects a true 5pp effect only ~28% of the time at the Holm bar, vs
  ~85% at n=20). 4 arms x 2 datasets x 20 seeds = 160 runs.
- Split: identical to the QOVT ablation -- fixed train/val split, val
  split 50/50 into val_sel (checkpoint selection) and test (held out,
  evaluated exactly once at the selected epoch), split_seed=0.
- Recipe: AdamW lr=1e-3 (single uniform LR, all params, all arms),
  wd=1e-4, CosineAnnealingWarmRestarts, **48 epochs** (raised from the
  QOVT ablation's 30 because of the recipe-confound concern disclosed in
  QOVT_PAPER.tex Limitations -- all four arms here are within 2% of each
  other in total size, so recipe choice cannot differentially
  under-train one arm the way it may have there; 48 not 50 so training
  ends exactly at a cosine-restart cycle minimum).
- Metric: held-out test macro OVR AUC.
- Per-run diagnostics logged: circuit angle L2 drift and
  ||U(theta_final) - U(theta_init)||_F, so a null on tests 1-2 can be
  diagnosed as "structure doesn't help" vs "the angles never moved"
  (near-zero init makes U approximately the CNOT-ring permutation, which
  the permutation-equivariant backbone can absorb at t=0).

## Pre-registered hypotheses and tests

Statistical machinery identical to the QOVT ablation: paired Wilcoxon
signed-rank across the 20 seeds, Holm-Bonferroni over the primary
family, practical-significance bar = Holm p<0.01 AND |mean delta|>=2pp
AND >=16/20 seeds agreeing in sign.

**Power statement (committed before running):** using the 08 ablation's
observed paired-diff SDs as the anchor, this design has ~85% power for a
true 5pp effect (sdD=5pp) and remains underpowered for effects near the
2pp practical bar when sdD is large. Consequently a null result licenses
only "no effect of >=5pp was detectable under this design" -- NOT "the
effect is zero".

**Primary family (Holm-corrected over these 4):**

| # | test | direction | what it isolates |
|---|---|---|---|
| 1 | quantum > scramble, Model_II | one-sided | circuit bit-structure (THE falsification test) |
| 2 | quantum > scramble, Model_III | one-sided | same |
| 3 | quantum > conv, Model_II | one-sided | inductive bias vs learned embed at low N |
| 4 | quantum > conv, Model_III | one-sided | same |

**Secondary (reported, uncorrected, no headline claims):**
- quantum vs dct, both datasets, TWO-sided (dct is the classical analog;
  neither direction has a prior claim -- if quantum ≈ dct, the honest
  conclusion is "the useful bias is spectral, not quantum").
- scramble vs conv, both datasets, two-sided (context).

## Interpretation commitments (written before seeing data)

- If tests 1-2 fail (no detectable quantum > scramble): given the power
  statement above, the licensed conclusion is "no circuit-structure
  effect of >=5pp is detectable in this design". The angle-drift
  diagnostic further splits this: if U barely moved from init, the test
  was uninformative (circuit never engaged); if U moved substantially
  and still no gap, that is genuine evidence against the structure
  mattering at this effect size, and the "quantum structure helps"
  framing must not be used in any project claim unless a
  higher-powered replication later finds it.
- Equivalence claims (e.g. "quantum ≈ dct") will NOT be made from a
  non-significant difference test. Committed rule: "quantum ≈ dct" may
  be declared only if the 90% CI of the paired mean delta lies entirely
  within ±2pp (TOST-style); otherwise the comparison is reported as
  inconclusive.
- If 1-2 pass but quantum ≈ dct (by the CI rule above): the winning
  ingredient is the spectral inductive bias, achievable classically at
  ~zero params; the honest framing is "the quantum circuit is a
  hardware-executable way to get a spectral embed", not "quantum beats
  classical".
- If 1-2 pass AND quantum > dct: the learnable circuit adds something
  beyond fixed DCT -- the strongest available result, still to be
  replicated at a second N before any strong claim.
- No cell will be dropped or re-run after seeing partial results. All
  tests reported regardless of outcome.
- Known dataset caveat carried over from the diagnostic: the spectral
  separability shows a comb-like period-4 structure that may partly be
  simulation grid artifacts; it affects all arms equally but any
  eventual writeup must disclose it.

## Results

*(to be filled in after the jobs complete; nothing above this line will
be edited after data exists)*
