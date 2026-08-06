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

## Results

*(to be filled in after the job completes — nothing above this line will be
edited once seeds start running)*
