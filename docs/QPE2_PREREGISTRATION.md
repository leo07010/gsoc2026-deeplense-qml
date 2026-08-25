# Pre-registration: QPE-2 -- Spectrally-Initialized Quantum Patch Embedding

**Date committed: 2026-08-19, BEFORE any sweep run of this experiment.**
Only unit tests and 2-epoch smoke runs existed when this was committed.

## Background: the derivation this design comes from

QPE-1 (`docs/QPE_PREREGISTRATION.md`, results in commit history) refuted
the bare circuit-structure hypothesis: a randomly-initialized RY+CNOT
patch embed ties its scrambled control, and the fixed 2D DCT is the best
arm at N=500. Two mathematical facts frame what remains winnable on
these datasets at low N:

1. **Same-function-class reparameterizations can only differ by
   optimization-conditioning effects** (the downstream learned MHA
   projections absorb any fixed orthogonal embed), which QPE-1 measured
   at the 1-2pp scale. To get more, the arms must differ in *prior*,
   not just chart.
2. **At extreme low N, estimation error dominates**, and the winner is
   "correct prior + smallest learnable correction". The correct prior
   is known (Step-0 diagnostic: intra-patch spectral, concentrated in
   specific bands the fixed DCT does not privilege), and the smallest
   correction space available that is also NISQ-executable is the
   48-angle RY+CNOT chart.

QPE-2 therefore composes  **U = R(params) @ T_dct**: every refinement
arm starts AT the spectral prior and differs only in the chart used to
refine it. The remaining quantum question -- made falsifiable here --
is whether the circuit chart is the best low-N way to learn that
refinement, against classical charts given MORE parameters.

## Arms (byte-identical classical ViT backbone, 203,459 params,
bit-identical init per seed; embed params include the scalar gain)

| arm | refinement chart R | embed params | init |
|---|---|---|---|
| quantum | RY+CNOT circuit (48 angles) | 49 | **exactly 0** -> R0 = CNOT-ring permutation P^8 (a pure channel relabeling of the DCT) |
| skew48 | Cayley chart restricted to 48 fixed skew coordinates -- the **param-matched** classical chart | 49 | 0 -> R0 = I exactly |
| butterfly | Givens butterfly (192 angles; QOVT's unstable classical chart, verbatim) | 193 | 0 -> R0 = I exactly |
| cayley | full skew-symmetric Cayley chart (2,016 params; 42x quantum) | 2,017 | 0 -> R0 = I exactly |
| dctfix | R = I frozen (the QPE-1 winner -- the bar) | 1 | -- |
| conv | learned Conv2d, no prior (context) | 4,160 | default |

Two review-driven design points, both settled before running:

- **Zero-init for the quantum arm** (QPE-1 used `0.01*randn`). At
  theta=0 the circuit is exactly the CNOT-ring permutation, so every
  refinement arm starts at an orthogonal relabeling of the same
  spectral prior and the arms differ *only* in chart, as claimed.
  Gradients at theta=0 are nonzero for all 48 angles (unit-tested), so
  this is not a dead start. The permuted-vs-exact-identity difference
  is immaterial because the backbone init distribution is exchangeable
  under channel permutation (cls=0, pos iid, LayerNorm affine 1/0, MHA
  weights iid), i.e. quantum's starting point is *distributionally*
  identical to dctfix's.
- **skew48 exists because otherwise the headline is unfalsifiable.**
  With only quantum(48) vs cayley(2016) in the primary family, a
  quantum win at N=100 is more parsimoniously explained by "48
  parameters generalize better than 2016 on 300 images" -- ordinary
  classical statistics -- than by anything about the circuit chart.
  skew48 is a classical chart with the *same* parameter count and the
  same init, so `quantum > skew48` is the test that actually isolates
  the chart. Its 48 coordinates are drawn once from a fixed seed and
  are identical in every run.

The quantum arm remains a legitimate gate-model construction: the fixed
DCT block is itself circuit-implementable (quantum DCT via QFT,
Klappenecker & Roetteler 2001), so R(theta) @ T_dct is a circuit
composition; we simulate it exactly as in all prior rounds.

## Protocol

- Datasets: Model_II, Model_III. **N in {100, 250}/class** -- the regime
  where the Caro bound is non-vacuous for T~50 and vacuous for the
  classical learned arms, and where QPE-1 already saw conv/sham
  instability precursors at N=500.
- Seeds 42-61 (20). 6 arms x 2 N x 2 datasets x 20 seeds = 480 runs.
- Split: identical to QPE-1/QOVT (fixed train/val; val split 50/50 into
  val_sel + held-out test, split_seed=0, test touched once).
- Recipe (all arms identical): AdamW lr=1e-3, wd=1e-4,
  CosineAnnealingWarmRestarts (T_0=32, ends at a cycle end), 96 epochs,
  batch 32 (N=100 is only 300 images), no augmentation.
- Metric: held-out test macro OVR AUC. Per-run diagnostic:
  ||R_final - R_init||_F for every refinement arm.

## Pre-registered hypotheses and tests

Paired one-sided Wilcoxon across 20 seeds; Holm-Bonferroni over the
primary family; practical bar = Holm p<0.01 AND |mean delta|>=2pp AND
>=16/20 seeds.

**Primary family (Holm over 6), all at N=100 (theory says the effect
grows as N shrinks, so N=100 is the pre-committed primary regime):**

| # | test | what it isolates |
|---|---|---|
| 1 | quantum > dctfix, Model_II | does ANY learnable refinement beat the fixed prior at low N |
| 2 | quantum > dctfix, Model_III | same |
| 3 | quantum > skew48, Model_II | **circuit chart vs a param-matched classical chart (48 vs 48)** -- the test that isolates the chart itself |
| 4 | quantum > skew48, Model_III | same |
| 5 | quantum > cayley, Model_II | circuit chart vs the full-dimensional classical chart (42x params) |
| 6 | quantum > cayley, Model_III | same |

**The headline claim "quantum wins at small data on this data" is
declared ONLY if tests 1-4 all pass the practical bar** (5-6 are
reported and expected to be easier, since cayley is the
over-parameterized arm; they are not sufficient on their own).
Passing 1-2 but not 3-4 means adaptation helps but a param-matched
classical chart does it just as well -- a classical result, not a
quantum win. Passing 5-6 without 3-4 means only "fewer parameters
generalize better at N=100", which is ordinary statistics and will be
reported as such.

**Secondary (reported, uncorrected):**
- The same six comparisons at N=250.
- quantum vs butterfly, both N, both datasets, one-sided quantum >
  (prior evidence: the butterfly chart is unstable in this codebase) --
  the chart-conditioning mechanism test.
- quantum vs conv, both N, both datasets (context).
- Equivalence statements only via the TOST-style rule: 90% CI of the
  paired mean delta entirely within +/-2pp.

## Power statement

Same anchor as QPE-1 (observed paired-diff SDs 3-17pp): n=20 gives ~85%
power for a true 5pp effect at sdD=5pp. At N=100 per-seed variance will
be larger than at N=500; a null licenses only "no effect >= ~5pp
detectable", not "no effect".

## Interpretation commitments (written before seeing data)

- If tests 1-2 fail: learnable refinement adds nothing detectable over
  the fixed spectral prior at low N; the honest project-wide conclusion
  becomes "the classical fixed DCT embed is the best known method here,
  full stop", and no further quantum-embedding rounds will be proposed
  on these datasets.
- If 1-2 pass but 3-4 fail: refinement is real but chart-agnostic --
  a classical result; the quantum arm's only remaining distinction is
  NISQ executability at equal accuracy, and it will be framed exactly
  that way.
- If 5-6 pass but 3-4 fail: the effect is parameter count, not chart.
  This will be reported as "at N=100, a 48-parameter refinement beats a
  2016-parameter one" -- a classical-statistics observation with no
  quantum content.
- If tests 1-4 all pass: the pre-registered small-data quantum win --
  reported with the explicit caveat that it is one round, on simulated
  data, and requires independent replication (different N or fresh
  seeds) before any strong claim.
- R_frob_drift will be used to diagnose any null (chart never moved vs
  moved-but-no-gain).
- No cell dropped or re-run after seeing partial results; all tests
  reported regardless of outcome.
- Carried-over caveat: the Step-0 spectral separability shows comb-like
  period-4 structure that may partly be simulator grid artifacts;
  affects all arms equally.

## Design note: circuit-manifold expressivity probe

A side probe (scratchpad `fit_circuit_to_dct.py`, running at commit
time) fits U(theta) directly to the 2D DCT to measure how close the
bare 48-angle manifold can get. Its result does not gate this
experiment (the composition design needs no fit) but will be reported
as context on the ansatz's expressivity.

## Results

Jobs 291060 (model_II) + 291061 (model_III), 2026-08-22, 1h09m each.
**480/480 runs completed, zero failures** (`failed_runs_*` empty). Raw
per-seed data: `experiments/11_qpe2_spectral/results_qpe2_model_*.jsonl`;
analysis `analyze_qpe2.py`, full output `qpe2_analysis_output.txt`.

**Arm means (n=20 seeds, held-out test AUC; parenthesized figure is the
population std, ddof=0):**

| arm (embed params) | N=100 Model_II | N=100 Model_III | N=250 Model_II | N=250 Model_III |
|---|---|---|---|---|
| skew48 (49) | **0.8542** (0.031) | **0.8866** (0.017) | **0.9292** (0.006) | **0.9349** (0.005) |
| dctfix (1) | 0.8516 (0.030) | 0.8843 (0.019) | 0.9287 (0.006) | 0.9335 (0.005) |
| butterfly (193) | 0.8488 (0.034) | 0.8819 (0.025) | 0.9275 (0.007) | 0.9341 (0.006) |
| quantum (49) | 0.8379 (**0.058**) | 0.8788 (**0.041**) | 0.9283 (0.006) | 0.9337 (0.007) |
| cayley (2017) | 0.8339 (0.044) | 0.8822 (0.013) | 0.9273 (0.009) | 0.9338 (0.006) |
| conv (4160) | 0.7302 (0.041) | 0.6872 (0.066) | 0.8940 (0.021) | 0.8950 (0.034) |

**Primary family (N=100, one-sided, Holm over 6) -- all six fail, five
of them with the sign reversed:**

| # | test | delta | pos | raw p | Holm | verdict |
|---|---|---|---|---|---|---|
| 1 | quantum > dctfix, II | -0.0137 | 9/20 | 0.806 | 1.000 | reversed |
| 2 | quantum > dctfix, III | -0.0056 | 12/20 | 0.435 | 1.000 | reversed |
| 3 | quantum > skew48, II | -0.0163 | 7/20 | 0.918 | 1.000 | reversed |
| 4 | quantum > skew48, III | -0.0079 | 11/20 | 0.565 | 1.000 | reversed |
| 5 | quantum > cayley, II | +0.0039 | 10/20 | 0.273 | 1.000 | coin flip (see below) |
| 6 | quantum > cayley, III | -0.0035 | 12/20 | 0.205 | 1.000 | reversed |

**HEADLINE: NO.** Tests 1-4 do not pass; the pre-registered small-data
quantum win is not observed, and the point estimates run the wrong way.
The failure is not marginal: the largest delta anywhere in the family
is +0.0039, against a pre-committed gate of +0.02. Test 5, the only
positive one, is 10/20 seeds with a median paired difference of
+0.0001 -- a coin flip, not a direction.

**Chart-drift diagnostic -- every chart moved, so the nulls are
informative:** R_frob_drift means (against ||R||_F = 8) quantum
0.70-0.84, skew48 1.22-1.40, butterfly 1.52-1.79, cayley 3.65-3.94.
No arm was frozen at its initialization.

**Secondary, N=250:** all six comparisons are null (|delta| <= 0.0012,
raw p 0.46-0.86), and the TOST rule declares quantum EQUIVALENT to both
dctfix and skew48 on both datasets (all four 90% CIs inside +/-2pp,
widths ~0.005). At N=250 the choice of chart -- or whether to refine at
all -- simply does not matter.

**Secondary, quantum vs butterfly:** null at every cell (|delta| <=
0.011, all p >= 0.32). Unlike the QOVT ablation, the butterfly chart
does not fail here; composing it with the DCT prior appears to remove
the instability that made it collapse in QOVT.

### Verdict per the pre-committed interpretation rules

- **Tests 1-2 fail, with all charts demonstrably moving.** The
  committed reading applies verbatim: *learnable refinement adds
  nothing detectable over the fixed spectral prior at low N; the
  classical fixed DCT embed is the best known method here, full stop,
  and no further quantum-embedding rounds will be proposed on these
  datasets.*
- The quantum arm is not merely equal but, **among the orthogonal
  arms, the least stable** at N=100: its per-seed std (0.058 / 0.041)
  is roughly double dctfix's (0.030 / 0.019), with a worst seed at
  0.681. (conv is less stable still on Model_III, std 0.066, but conv
  is the no-prior baseline, not an orthogonal chart.) This runs opposite to
  the QOVT-era "orthogonal circuits are more stable" observation --
  there, every comparison arm was unconstrained; here every arm is
  orthogonal, and among orthogonal charts the circuit is the
  worst-conditioned, not the best.
- Bonus: skew48 is nominally the top arm in all four cells, but its
  margin over dctfix (0.04-0.26pp) is far inside noise and was never a
  pre-registered test; it is reported as an observation, not a finding.

### The one large effect in this round is classical

The fixed spectral prior (the **dctfix** arm) beats the learned conv
embedding by a margin an order of magnitude larger than any quantum
effect ever measured in this project: **+12.1pp (Model_II) and +19.7pp
(Model_III) at N=100, both 20/20 seeds**, still +3.5pp (20/20) and
+3.9pp (19/20) at N=250. (The quantum arm's own margins over conv --
+10.8pp / +19.2pp / +3.4pp / +3.9pp -- are similar because every
non-conv arm composes with the same DCT prior; the effect belongs to
the prior, not the chart.) Combined with QPE-1 (DCT > conv by 2.0/6.1pp
at N=500), the low-data story on these datasets is entirely about
having the right fixed spectral prior, and not at all about quantum
circuits.
