# QOVT Ablation — Pre-Registration

Written and committed **before** `experiments/08_qovt_ablation/train_qovt_ablation.py`
is submitted to SLURM. Purpose: this project's own audit (see
`docs/BENCHMARK_DATABASE.csv` row 29 and the QOVT inventory,
2026-08-06) found that its previous "pre-registration" for the QVF-Hybrid
significance campaign could not be independently verified from git history,
and that its confirmatory sample partly overlapped the discovery sample that
motivated the test. This document exists so that trap can't repeat here: the
test, the cells, the seeds, and the correction method are fixed now, in a
file whose commit timestamp precedes every run, and the results section below
will be filled in only after the job completes — nothing here will be edited
post-hoc except to append that section.

## Background

An adversarial audit of the QOVT training-script family (8 scripts in
`/home/leo07010/mae-lensing/`) found:
1. No held-out test set — all scripts report `max(val_AUC)` over training,
   using the same split for both model selection and the reported metric.
2. No parameter-matched classical control (existing sham is 33k params vs
   quantum's 384 — an 85x mismatch).
3. Quantum's attention params get a 10x higher LR than every other arm's
   attention params (an unequalized confound, not a considered ablation).
4. The only cell ever run at n=3 seeds (Model_II, full data) shows sham
   *beating* quantum (0.9911 vs 0.9852) — the more rigorously tested cell
   already points against the quantum-wins direction.
5. The one gap that exceeds this project's own established noise floor is
   RY+CNOT quantum 0.9144 vs sham 0.8812 at Model_II, N=500 (+0.033,
   n=1) — but this project's own multi-seed campaigns have shown single-seed
   effects here shrink 6-11x under proper testing (see
   `qvf-audit-findings-20260731` memory), so this is the cell most likely to
   either replicate or evaporate, and therefore the one worth spending
   compute on.

`train_qovt_ablation.py` fixes issues 1-3 (see its module docstring for
exact mechanism) and adds a fourth arm ("matched": rank-1 low-rank U/V,
1024 params — the leanest linear-algebra classical control achievable,
disclosed as not an exact 1:1 match to quantum's 384).

## Pre-registered hypotheses and test cells

**Primary tests (4, pre-specified, no others will be added after seeing data):**

| # | Comparison | Dataset | N/class |
|---|---|---|---|
| 1 | quantum > matched (rank-1 low-rank, 1024 params) | Model_II | 500 |
| 2 | quantum > matched | Model_III | 500 |
| 3 | quantum > sham (unconstrained Linear, 33k params) | Model_II | 500 |
| 4 | quantum > sham | Model_III | 500 |

**Secondary (reported, not used for the pass/fail claim):** quantum vs
classical (full MHA, 66k params), same 2 cells — included for completeness
since the reference paper (Cherrat et al.) compares against classical ViT
attention, but not part of the primary claim because "classical" was already
established (BENCHMARK_DATABASE.csv row 1) to not be the right floor to
chase parameter-efficiency claims against.

## Design

- **n = 10 seeds**, seeds 42-51 (matches this project's existing convention).
- **Fixed test split**: `split_seed=0` for every run — same held-out test
  set across all seeds/arms/datasets, carved once from the original val set
  (50/50 into val_sel/test), never touched during training or model
  selection.
- **Arms**: quantum, matched, sham, classical (4) x seeds (10) x datasets (2)
  x N=500 = **80 runs**.
- **Test**: paired one-sided Wilcoxon signed-rank test (quantum's per-seed
  test AUC vs the comparison arm's per-seed test AUC, same seed paired),
  Holm correction across the 4 primary tests.
- **Alpha**: 0.01 one-sided, matching the stricter bar this project settled
  on after its own significance campaign was found to not survive a looser
  bar (see qvf-audit-findings-20260731).
- **No optional stopping.** All 4 primary-test results will be reported
  regardless of outcome, including if quantum loses or ties. No cell will be
  dropped, re-run with different seeds, or extended after seeing a partial
  result.

## What would make this a *positive* result

Per this project's own established practical-significance bar (from the
QVF-Hybrid campaign): a raw Wilcoxon p < 0.01 alone is not sufficient given
the demonstrated seed-shrinkage pattern (6-11x). A cell will be treated as a
genuine candidate finding only if it clears Holm-corrected p < 0.01 **and**
the point-estimate margin is >= 2% AUC **and** the sign is consistent in
>= 8/10 seeds. Anything short of that is reported as directional-only, same
standard applied to the (ultimately retracted) QVF-Hybrid claim.

## Run 1 (job 237915, 2026-08-06) — INVALIDATED, script bug found

The first sweep (80/80 runs completed) produced an implausibly large effect
(+23pp to +35pp AUC, quantum beating every control, Holm p=0.0039 on all 4
primary tests) — an order of magnitude larger than any real effect this
project has ever measured, which is itself the reason it was not accepted
at face value. Training-curve inspection
(`logs_run1_invalidated/{sham,matched}_model_II_seed42.out`) found the
cause: the script's own two-tier LR split (module docstring item #3)
applied `qlr=1e-2` to sham's `Linear(64,64)` and matched's low-rank U/V —
layers an order of magnitude larger than quantum's 48-param circuit that
`qlr` was calibrated for. `matched` never moved off chance (flat ~0.50 for
all 30 epochs); `sham` rose slightly then plateaued ~0.51-0.57. This was a
bug in the ablation script's attempted fix for the *original* QOVT family's
LR asymmetry (raw materials in `qovt_inventory.md` item 3) — my fix
generalized the wrong direction, applying a rate calibrated for a
tiny circuit onto much larger classical layers instead of removing the
asymmetry.

**Fix**: `train_qovt_ablation.py --qlr` now defaults to `None`, meaning
every parameter in every arm uses a single uniform `--lr` (default 1e-3,
unchanged). The two-tier split is now opt-in only (`--qlr <value>`), kept
for a possible future LR-sensitivity ablation, not used in the primary
comparison. Quick verification (15-epoch smoke test, Model_II N=500,
seed=42, uniform lr=1e-3): all four arms now show gradual, non-degenerate
learning curves (quantum 0.67, sham 0.55, classical 0.53, matched 0.55 by
epoch 15) — none flat, none diverging.

Raw invalidated results preserved at
`results_ablation_RUN1_INVALIDATED_qlr_bug.jsonl` and
`logs_run1_invalidated/` for audit purposes. **Run 1's numbers must never be
cited as a QOVT result.** Same pre-registered design (4 primary tests, 10
seeds 42-51, Holm-corrected paired one-sided Wilcoxon, alpha=0.01,
practical-significance bar) reruns unchanged in Run 2 below — only the
script's LR bug is fixed, nothing about the test itself.

## Run 2 (corrected script, job 251067, 2026-08-11) — Results

80/80 runs completed (4 arms × 2 datasets × 10 seeds × N=500). Full raw
results: `results_ablation.jsonl`. Analysis exactly as pre-registered:
`analyze_results.py`, paired one-sided Wilcoxon signed-rank per cell,
Holm-Bonferroni step-down over the 4 primary tests, α=0.01, practical bar =
Holm p<0.01 **and** |Δ|≥2% **and** consistent sign in ≥8/10 seeds.

**Primary tests (Holm-corrected):**

| # | Test | mean(quantum) | mean(control) | Δ | pos/10 | raw p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Q > matched, Model_II | 0.8855 | 0.5490 | **+0.3366** | 10/10 | 0.0010 | 0.0039 | **CANDIDATE FINDING** |
| 2 | Q > matched, Model_III | 0.8927 | 0.5700 | **+0.3227** | 10/10 | 0.0010 | 0.0039 | **CANDIDATE FINDING** |
| 3 | Q > sham, Model_II | 0.8855 | 0.7290 | **+0.1566** | 10/10 | 0.0010 | 0.0039 | **CANDIDATE FINDING** |
| 4 | Q > sham, Model_III | 0.8927 | 0.8188 | +0.0739 | 7/10 | 0.0137 | 0.0137 | directional only (fails Holm α=0.01 and the 8/10 bar) |

**Secondary (reported, not part of the primary claim, no correction):**

| # | Test | mean(quantum) | mean(control) | Δ | pos/10 | raw p |
|---|---|---|---|---|---|---|
| 5 | Q > classical (MHA), Model_II | 0.8855 | 0.7492 | +0.1363 | 9/10 | 0.0049 |
| 6 | Q > classical (MHA), Model_III | 0.8927 | 0.6991 | +0.1937 | 10/10 | 0.0010 |

**All four arms' means (both datasets):**

| Arm | Model_II mean±std | Model_III mean±std | Params (attn / total) |
|---|---|---|---|
| quantum (RY+CNOT) | 0.8855±0.0703 | 0.8927±0.0545 | 384 / 142,467 |
| matched (rank-1 low-rank) | 0.5490±0.0209 | 0.5700±0.0243 | 1,024 / 143,107 |
| sham (unconstrained Linear) | 0.7290±0.1526 | 0.8188±0.1072 | 32,768 / 174,851 |
| classical (full MHA) | 0.7492±0.0746 | 0.6991±0.1191 | 65,536 / 207,619 |

**Reading these results:**

- **3 of 4 primary tests clear the pre-registered bar.** Quantum beats the
  rank-1 "matched" control decisively and beats the unconstrained-Linear
  "sham" decisively on Model_II. This is the first QOVT result in this
  project's history to survive a 10-seed, Holm-corrected, pre-registered
  test — none of the prior single-seed QOVT numbers (including the original
  id-23 butterfly-variant numbers) has been tested this way.
- **Test 4 (Q > sham, Model_III) does not clear the bar.** Its point-estimate
  margin (+7.39pp) does clear the pre-registered ≥2% threshold on its own —
  it fails on the *other two* criteria: Holm p=0.0137 (>0.01) and only 7/10
  seeds positive (<8/10 required). Sham on Model_III is genuinely noisy
  (std=0.1072, the 3rd-highest of the 8 arm/dataset cells in this sweep —
  sham/Model_II is noisier still, at std=0.1526), which is consistent with
  the inconsistent sign across seeds. This is reported as a negative result
  for this cell, not rounded up.
- **The "matched" wins (tests 1–2) are the least surprising of the four.**
  A rank-1 linear factorization is a genuinely weak function class (it can
  only represent rank-1 bilinear maps) — losing to it is a low bar to clear,
  and the pre-registration document disclosed this control was not an exact
  parameter match (1,024 vs quantum's 384) precisely so this result isn't
  overread as evidence against a stronger control. **The sham comparison
  (tests 3–4) is the more informative test**, since sham's 32,768-param
  unconstrained Linear is a materially more expressive function class,
  despite still being linear — and there, the result is genuinely mixed:
  significant on Model_II, not on Model_III.
- **Not yet a "confirmed" finding by this project's own standard.** This is
  a single pre-registered round at one N (500) on two datasets. Per this
  project's own seed-shrinkage precedent (§ this doc's "Background," item
  5; see `qvf-audit-findings-20260731`), a first significant result — even
  a large, Holm-corrected one — should be treated as a strong candidate for
  replication, not a closed case. The three tests that cleared the bar
  (15.7–33.7pp) are an order of magnitude larger than the QVF-Hybrid effect
  that shrank 6–11× and ultimately failed replication — reassuring, but not
  a substitute for actually replicating it, and it does not rescue test 4
  (+7.4pp, did not clear the bar). **Recommended before calling this
  "confirmed":** rerun at a second N (e.g. N=1000 or N=250) with a fresh
  seed range (e.g. 60–69) as a genuinely independent confirmatory sample,
  the same discipline that the QVF-Hybrid campaign got right in its design
  and wrong in its execution (discovery/confirmatory seed overlap).
- **What this result does NOT yet establish:** whether the effect is
  specific to the RY+CNOT circuit or would also appear with the RBS/Givens
  butterfly variant (id 23 in the benchmark database) — those are different
  circuits, with the butterfly's quantum-arm attention parameters (1,536
  trainable angles) outnumbering RY+CNOT's (384) by **4×**, despite both
  landing at nearly the same *total* model size (~143k either way, since
  the CNN backbone dominates the count). This ablation has only tested
  RY+CNOT. See the project report §3.6 for why these are kept distinct.

Recorded in `docs/BENCHMARK_DATABASE.csv` as a new row (id 33), verdict
`quantum_wins (directional, single pre-registered round, not yet
replicated)` rather than a plain `quantum_wins`, to keep this result's
actual evidentiary weight visible at a glance.

---

## Round 2 pre-registration: Butterfly circuit (closing the reference-paper
## Table 1 gap) — written and committed BEFORE any butterfly run happens

Leo asked to "close the reference-paper structure gap" (Cherrat et al.'s
Table 1: Pyramid / X / Butterfly circuit topologies). Investigation found
this project cannot faithfully reproduce Pyramid or X from the paper alone
— their exact qubit-connectivity pattern is shown only in a circuit diagram
figure, which plain-text PDF extraction cannot recover; only the parameter-
count formulas (Table 1) are available in text. Leo chose (2026-08-11,
AskUserQuestion) to scope this round to **Butterfly only** — the one
topology this project already has a verified-correct implementation of
(`train_qovt.py`'s `RBSLayer`, reused verbatim below, not reinvented) — and
defer Pyramid/X to a future round that either finds the paper's actual
circuit figure or clearly labels a from-scratch design as this project's own
construction, not a reproduction.

**Implementation, verified before any training run:**
`ButterflyLayer` in `train_qovt_ablation.py` is a byte-for-byte copy of
`/home/leo07010/mae-lensing/train_qovt.py`'s `RBSLayer` (the circuit that
produced id=23's existing single-seed numbers) — so this round tests the
SAME circuit id=23 already reports on, not a new guess. Verified by direct
computation before running anything:
- Orthogonality: `max|W W^T - I| = 4.17e-07` for a random-angle instance —
  confirms the circuit genuinely produces an orthogonal matrix.
- Parameter count: 192 angles per instance, exactly matching the paper's
  `(D/2)*log2(D)` formula at D=64 (`(64/2)*log2(64) = 32*6 = 192`).
- Full-model count when wired into this ablation's QOVT class: attn_params
  1,536, total 143,619 — **exact match to id=23's already-published
  numbers** (143,619 total, 1,536 angles), confirming this is the same
  circuit under the same overall architecture, not a different model that
  happens to have a similar name.
- Smoke test (2 epochs, N=50, Model_II): ran to completion with no errors
  (`test_AUC=0.4905`, chance-level as expected for a 2-epoch smoke run).

**Design.** Since `matched`/`sham`/`classical` controls do not depend on
which quantum circuit is being tested (each arm is trained as a fully
independent model — verified by inspecting `QOAttention.__init__`, which
only branches on `mode`, never referencing the quantum arm's circuit
choice), this round **reuses the existing matched/sham/classical rows from
`results_ablation.jsonl`** (job 251067, same seeds, same data splits, same
protocol) rather than re-running them. Only the quantum arm is newly run,
with `--circuit butterfly`, same N=500, same 2 datasets, same seeds 42-51,
same split_seed=0 — **20 new runs** (vs. 80 for the RY+CNOT round).

**Pre-registered primary tests (4, Holm-corrected, same α=0.01, same
practical bar as round 1 — Holm p<0.01 AND |Δ|≥2% AND ≥8/10 seeds):**

| # | Comparison | Dataset |
|---|---|---|
| 1 | butterfly > matched | Model_II |
| 2 | butterfly > matched | Model_III |
| 3 | butterfly > sham | Model_II |
| 4 | butterfly > sham | Model_III |

**Secondary (reported, not part of the primary claim):**
- butterfly > classical, both datasets (same rationale as round 1: not the
  right floor to chase a parameter-efficiency claim against).
- **butterfly vs. RY+CNOT, both datasets, two-sided** (paired by seed —
  both circuits were trained under the identical protocol/split/seed) — this
  is the actual "which topology wins" question Table 1 raises, and is
  explicitly two-sided since neither circuit has a prior directional claim
  over the other in this project's own results.

No cell will be dropped or re-run after seeing a partial result. All 4
primary + all secondary tests will be reported regardless of outcome.

### Round 2 — Results

*(to be filled in after the job completes)*
