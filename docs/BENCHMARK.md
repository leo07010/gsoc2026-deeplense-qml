# Benchmark Database — Quantum vs Classical on DeepLense

This is a structured, machine-readable version of this project's exhaustive
quantum-vs-classical experiment log. Where `docs/EXPERIMENTS_MASTER.md` is a
narrative lab notebook (chronological, prose-heavy), `BENCHMARK_DATABASE.csv`
is one row per distinct method/architecture tried, with matched-control
numbers, deltas, and an explicit verdict — built so other researchers can
scan, filter, or reuse individual results without reading the full log.

Every quantum method here is paired with a **capacity-matched classical
"sham"** control (same wrapper architecture, quantum circuit swapped for a
classical layer of matched dimensionality). `quantum − sham` is the only
clean test of a genuine circuit contribution; comparisons against oversized
or pretrained baselines (MAE-ViT) are reported too but kept separate, since
they conflate the circuit with data/parameter advantages.

## How to read the CSV

`docs/BENCHMARK_DATABASE.csv`, 33 rows, one per method:

| Column | Meaning |
|---|---|
| `id` | Row number, stable reference for citing a specific method |
| `method_name` | What it's called in this project |
| `category` | Discriminative head / feature-extractor / kernel / generative / anomaly / metrology, etc. |
| `mechanism_one_line` | Where the circuit sits and what it replaces |
| `dataset(s)` | Which DeepLense variant(s) it was tested on (Model_I/II/III/V, Dataset1, CommonTest) |
| `N_range_tested` | Training-set sizes swept |
| `quantum_params` / `classical_control_params` | Parameter counts — always check these are close before trusting a delta |
| `best_quantum_metric` / `best_classical_or_sham_metric` | The headline numbers (mostly macro-OVR AUC) |
| `delta` | quantum − control, with direction and any caveats about consistency |
| `n_seeds` | How many random seeds back the number — **most rows are single-seed**, treat accordingly |
| `statistically_tested(Y/N)` | Whether a formal significance test (not just a point estimate) was run |
| `verdict` | One of `quantum_wins`, `tie`, `quantum_loses`, `certificate_impossible`, `retracted`, `inconclusive` |
| `code_file` | Training script, where it exists in this repo checkout (many of the later scripts live only on the HPC cluster, not in this git history — noted per row) |
| `source_line_in_EXPERIMENTS_MASTER` | Line range in the full narrative log, for the complete story |
| `caveats` | The honest-science part — leakage warnings, non-convergence, noise-band calls, and (for row 29) the statistical audit correction |

`certificate_impossible` is a stronger claim than `quantum_loses`: it means a
computed geometric-difference certificate (Huang et al. 2021) proves no
quantum-kernel advantage can exist for **any** labeling of this data, not
just that none was found.

## The honest summary

Of the 33 rows, **31 are image-classification results** on DeepLense
strong-lensing data and **2 are genuinely different tasks**: row 12 (IQP
Born-machine generative model, tested on a synthetic bitstring benchmark,
not DeepLense images) and row 32 (quantum-sensor metrology, "QELP Track 2"
— estimating properties of quantum-sensor measurement data itself). Both
are kept in their own category so neither is ever read as an
image-classification result.

Among the 31 image-classification methods: **7 won outright** against their
matched sham (rows 6, 7, 19, 23, 26, 30, 33 — QVF-scratch amplitude/angle
encoding, LensPINN+QVF head, QOVT, the entanglement ablation, the
CNN-family-vs-MAE-paradigm result, and the pre-registered QOVT-ablation
result). **"Won outright" means beat its own matched sham/control — it does
not mean beat the best available classical model, and it does not mean
"confirmed."** Row 1 (a 93,763-param plain CNN with no quantum component at
all) Pareto-dominates the whole NAE-bottleneck family (rows 6, 7, 26, and row
29's retracted claim) and the QOVT-butterfly family (row 23) at nearly every
(dataset, N) — each of those rows' caveats now cross-references row 1
explicitly. Row 30's win is a "wins vs a different paradigm" claim already
honestly attributed to CNN inductive bias rather than the quantum component
(see its own caveats); row 19's "win" ties an internal unconstrained
classical-head control at its reported N; **row 33 is the first result in
this project's history to clear a pre-registered, Holm-corrected, 10-seed
bar** — 3 of its 4 primary tests pass, the 4th does not, and its own caveats
explicitly flag it as "not yet independently replicated," in keeping with
this project's rule that a single significant round (even a large one)
warrants a second independent confirmatory round before being called
settled. **10 tied** (no consistent edge either direction), **6 lost to
their classical control**, **2 were proven certificate-impossible** (quantum
kernels — no advantage can exist on this data for any labeling), **4 are
inconclusive** (both arms near chance, or a mixed/non-replicating pattern
with no clean call), and **2 were retracted** — one for a caught
label-leakage bug (row 13, QAE anomaly detection), and one — the project's
own former headline result — for failing an independent statistical audit
(row 29). The one theorem-backed quantum advantage that still stands (row
32) exists in a different task (quantum sensing), not in image
classification.

**The corrected result, in detail (row 29):** `EXPERIMENTS_MASTER.md` itself
reports a "STATISTICALLY SIGNIFICANT" module-level quantum-vs-sham effect at
Model_II, N=250 (p=0.0032, n=20, surviving its own pre-registered Holm-6
correction). An independent adversarial audit found this does not hold up:
the sham control had a bug (`train_qvf_opt.py:104`) that kept it from
reducing to the true classical floor, meaning a chunk of the reported gain
was the control's own handicap rather than a quantum effect; the p-value
fails multiple-comparisons correction by roughly one to two orders of
magnitude once corrected for the full search that identified this cell as
promising (not just the local confirmatory test); and splitting the sample
into original-discovery vs genuinely-new seeds shows the new-seeds-only half
is null (p=0.097). The corrected verdict is `retracted`, not `quantum_wins`
— documented in row 29's `caveats` field in full, with both the original
claim and the audit's counter-evidence laid out so readers can check the
reasoning themselves.

This repo's stated value is that **negative results are reported, not
buried** (see `README.md`). Retracting our own strongest-looking result once
an audit didn't hold up is that same principle applied to ourselves — it is
presented as a demonstration of the methodology working, not hidden or
soft-pedaled.

## Full narrative detail

- [`docs/EXPERIMENTS_MASTER.md`](EXPERIMENTS_MASTER.md) — the complete
  chronological lab log this database was built from, including every table,
  sweep, and reasoning trail summarized here.
- [`PROPOSAL.md`](../PROPOSAL.md) — project motivation and GSoC proposal
  context.
- [`docs/RESULTS.md`](RESULTS.md) — additional results detail.

## Caveats that apply project-wide

- Most rows are **single-seed** (`n_seeds` column) — treat any delta under
  ~0.5% as noise unless the row explicitly reports a multi-seed or
  statistical test.
- "Sham" always means a capacity-matched classical replacement for the
  quantum circuit, trained under the identical protocol — not a generic
  classical baseline.
- Several later-generation training scripts (`train_qvf_hybrid.py`,
  `train_qvf_opt.py`, `train_qvf_wide.py`, `train_qvf_tda.py`,
  `train_qvf_noent.py`, `qvf_plain_baseline.py`, `train_mae_cmp.py`, the
  `qelp_*.py` metrology scripts) are referenced by `EXPERIMENTS_MASTER.md`
  but are not present in this git checkout — the CSV's `code_file` column
  notes this per row rather than guessing a path.
