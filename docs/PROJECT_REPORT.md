# Project Report — Hybrid Quantum-Classical Methods for DeepLense Dark-Matter Substructure Classification

**Author:** leo07010 · **Project:** GSoC 2026, ML4SCI / DeepLense (QML track)
**Repo:** [leo07010/gsoc2026-deeplense-qml](https://github.com/leo07010/gsoc2026-deeplense-qml)
**Report date:** 2026-08-11

---

## 1. Executive summary

This project ran a systematic, sham-controlled search for a genuine quantum
advantage on the DeepLense dark-matter-substructure classification benchmark
(64×64 simulated strong-lensing images, 3-class: `axion` vortex / `cdm`
subhalo / `no_sub`). Across **33 distinct methods** spanning eight
architectural categories — discriminative heads, feature extractors,
kernels, generative models, anomaly detectors, equivariant layers, attention
mechanisms, and one quantum-sensing track — every quantum component was
paired with a capacity-matched classical control ("sham"), and every claim
was subjected to the same question: *does this survive scrutiny, not just a
single favorable run?*

The honest answer, after the full sweep: **exactly one image-classification
method in this project has produced a quantum-vs-own-sham gap that survived
a pre-registered, multi-seed significance test — and even that one is a
single, unreplicated round, reported as a candidate, not a settled result**
(§6, row 33). Of the other 30 image-classification methods, six
point-estimate "won" against some matched control — five of those (rows 6,
7, 19, 23, 26) at single-seed, never formally tested, and one of the five
(row 19, §3.2) only ties an internal unconstrained-classical control at its
reported N; the sixth (row 30) *was* pre-registered and multi-seed (3
seeds), but against a fixed Δ≥3%-and-3/3-positive threshold rather than a
significance test, and its own caveats attribute the win to CNN inductive
bias rather than the quantum component (§3.7); ten tied outright; six lost;
four were inconclusive (near-chance in every
arm, or a non-replicating pattern); two were proven *impossible* by a
closed-form certificate; and two were retracted after closer scrutiny
caught real flaws — a label-leakage bug, and, the project's own former
headline claim, a broken control plus an uncorrected multiple-comparisons
problem. Two results sit outside image classification entirely: a
generative model tested on a synthetic bitstring benchmark (§3.4, `loses`)
and one theorem-backed quantum-sensor-metrology advantage that stands
(§3.9).

Given that record, this project's forward-looking method is **QOVT
(Quantum Orthogonal Vision Transformer)** — chosen not because it had
already won, but because it was the one line of work whose comparison is
architecturally the cleanest (same ViT backbone, attention mechanism
swapped, not a bolt-on module). §6 reports the first payoff of that choice:
a pre-registered ablation that found a real, Holm-corrected signal (not yet
independently replicated) on the RY+CNOT variant of QOVT's attention
mechanism — the first result in this project's history to clear that bar.
§5 below explains
that choice in full; §6 reports the first round of that stress-testing,
including a methodological bug this project found and fixed in its own new
ablation code before trusting any number from it.

---

## 2. Research question and methodology

**Question:** does adding a quantum circuit to a classical vision model
improve classification of dark-matter substructure from strong-lensing
images — and if so, where in the architecture, under what training regime,
and is the effect real or an artifact of an unfair comparison?

**Standing methodology, applied to (almost) every experiment below:**

1. **Sham controls.** Every quantum component is paired with a
   capacity-matched classical replacement — same wrapper architecture, same
   training protocol, quantum circuit swapped for a classical layer of
   matched dimensionality. `quantum − sham` isolates the circuit's
   contribution from everything else (data, optimizer, architecture shape).
2. **A "null baseline."** Partway through the project, a plain 93,763-param
   CNN with no quantum component at all was benchmarked against the entire
   NAE-bottleneck quantum family and found to Pareto-dominate it at nearly
   every (dataset, N) — see row 1 and §4.1. This recontextualized every
   earlier sham-based "win" in that family: the sham itself had been
   handicapped by a shared bottleneck, not a true classical floor.
3. **Honest-evidence labeling.** Single-seed results are labeled as such;
   incomplete runs are labeled *incomplete*; leaked or otherwise invalid
   comparisons are labeled and, where the flaw is severe enough, retracted
   rather than kept with a footnote.
4. **Self-audit on the strongest claim.** When this project's best-looking
   result (QVF-Hybrid, a statistically "significant" p=0.0032 module-level
   quantum contribution) was put through an independent adversarial audit,
   it did not survive — see §4.2. That audit is itself part of this
   project's methodology, not an embarrassing footnote: catching your own
   best result being wrong, and saying so, is the standard the rest of this
   report holds every other claim to as well.

---

## 3. Complete method inventory

All 33 methods are cataloged with full detail — parameter counts, exact
metrics, seed counts, and caveats — in
[`docs/BENCHMARK_DATABASE.csv`](BENCHMARK_DATABASE.csv) and summarized in
[`docs/BENCHMARK.md`](BENCHMARK.md). The table below groups them by
category with the corrected verdict; ids refer to the CSV's `id` column.

### 3.1 Discriminative heads on frozen or fine-tuned features (ids 2, 3, 4, 9, 21, 24)

Quantum readout heads (gated fusion, cross-attention fusion, QCT, QVF, dual-
encoder FiLM, QONN) bolted onto features that are already discriminative
(frozen fine-tuned CLS features, AUC≈0.98 before any head is attached).
**Verdict: tie, uniformly.** With the feature ceiling already saturated, no
head — quantum or classical — can add information; this is corroborated by
the MAE paper's own finding that *frozen* MAE features are not linearly
separable (AUC 0.5365) while *fine-tuned* ones are — the discriminative work
happens in representation-shaping, not in the head. Testing a head-only
question on finished features cannot detect a circuit contribution even in
principle. One variant (id 21, QVF head on an MAE-pretrained ViT) is
`inconclusive` rather than `tie`: both arms collapsed to near-chance because
the MAE-ViT encoder itself can't be adapted at low data, an encoder problem
unrelated to the quantum question.

### 3.2 End-to-end feature extractors and physics-preprocessed heads (ids 5, 6, 7, 8, 19, 20, 22)

Where the circuit participates in *shaping* the representation from raw
pixels (or a physics-preprocessed version of them), rather than reading out
an already-finished one.

- **QVF-scratch (ids 6, 7)** and **LensPINN + QVF head (id 19) — the
  project's most consistent positive signal**, and the pair everything else
  in this category is measured against. QVF-scratch: CNN → learnable neural
  amplitude encoding → 8-qubit circuit → readout, trained end-to-end.
  Positive Q−sham at **21 of 22** sweep points (amplitude encoding), peak
  +0.1124 AUC at N=1500 on Model_II, shrinking toward a ±0.006 ceiling band
  as data saturates — the signature of a real inductive-bias effect rather
  than noise (which would flip sign, not shrink monotonically). Angle
  encoding (id 7) mostly replicates the direction but has a training-
  stability failure on Model_III at full data. Id 19 puts the same quantum
  readout head on physics-preprocessed (LensPINN log+Laplacian) features
  instead of raw pixels and finds the same direction, smaller magnitude
  (Q−sham +0.0014 to +0.0024, positive at all 3 sizes tested) — together,
  ids 6/7/19 are this project's evidence for where a quantum readout head
  helps. **Caveat that matters most:** every QVF-scratch point uses the
  shared 8-dim NAE bottleneck, and §4.1's null-baseline finding shows a
  plain CNN with *no* bottleneck beats this entire family outright at low
  N — so "quantum beats sham" is real but "quantum is competitive with the
  best available classical model" is not.
- **QCT-scratch (id 5)** — an end-to-end quantum *token* (not the readout-
  head pattern above): **ties**, no consistent edge.
- **QViT (id 8)** — quantum circuit inserted *mid-encoder* as an add-on
  (not a full attention replacement): **loses**, 0.962 vs 0.970.
- **QCNN / quanvolution (id 20)** — quantum *replaces* the CNN feature
  extractor: **loses badly** on Model_I (−0.11 to −0.21 AUC), ties on
  Model_II.
- **QMAE (id 22)** — amplitude-encoded head with no CNN front-end: **loses**
  by −0.012 to −0.050 across all tested configurations.

Cross-cutting pattern (the project's own "surrogate theorem"): **quantum
helps as a low-dimensional regularizing readout, and hurts or ties as a
high-dimensional feature extractor.** This pattern, observed independently
across ids 6/7/19 (readout, helps), 5/8 (add-on/token, ties or loses), 20
(replaces, loses badly), and 22 (loses), is one of the more load-bearing
structural findings in the whole project — it directly informs where QOVT
sits (§5).

### 3.3 Kernels (ids 10, 11)

Fidelity (id 10) and projected (id 11, PQK) quantum kernels, swept 8→16
qubits, tested via few-shot SVM (convex, no training-instability confound).
**Verdict: certificate-impossible** for both — a computed geometric-
difference certificate (Huang et al. 2021) proves no quantum-kernel
advantage can exist for *any* labeling of this data. The fidelity kernel's
gap to the advantage threshold *shrinks monotonically* as qubits increase,
5.7→2.7→2.3 (exponential concentration, getting worse with scale); the
projected kernel's gap is non-monotonic (11.5→12.3→10.1) but stays far below
the threshold at every qubit count tested. This is a stronger, closed-form
result — not "we didn't find one," but "one cannot exist here."

### 3.4 Generative / anomaly (ids 12, 13, 14)

- **IQP Born machine (id 12)**: **loses** to every classical generative
  baseline tested (matched Ising, autoregressive, mixture-of-Bernoullis) on
  held-out NLL — interference actively hurts on these low-order natural
  latents.
- **QAE anomaly detection**: the headline early result (id 13, 72-param
  quantum AE matching a 2,308-param classical AE at AUC≈0.996) turned out to
  be measured on **label-leaked** features (a label-fine-tuned encoder, not
  a self-supervised one) — **retracted**. Re-run on genuinely leakage-free
  SSL features (id 14), every learned arm — quantum and classical alike —
  collapsed to AUC 0.44–0.57, while a *zero-parameter* Mahalanobis baseline
  on the same features reached 0.86–0.91. **Quantum loses**, but so does
  every learned classical alternative; the real finding is a leakage warning
  for the field, not a quantum-specific one.

### 3.5 Equivariant / robustness / multi-view / few-shot (ids 15, 16, 17, 18)

- **REQAE (id 18)**, a C4-rotation-equivariant quantum layer: a large
  low-data win (+0.11 to +0.18 AUC at N=50–100) — but a matched
  *classical*-equivariant layer wins by the same margin. **The win is the
  symmetry, not the circuit** (quantum ≈ classical-equivariant throughout;
  the equivariance-over-plain gain itself collapses from +0.18 at N=50 to
  +0.0045 at full data, and at full data quantum loses to classical in
  *both* the equivariant and the plain arm). **Verdict: quantum_loses** in
  the sense that matters — no circuit-specific contribution once the
  symmetry is controlled for.
- Robustness battery, multi-view fusion (M=1→8), and few-shot (2
  architectures × 2 datasets × 5 seeds): all **tie**, no consistent edge in
  either direction.

### 3.6 Attention-mechanism replacement — QOVT (id 23)

Cherrat et al.'s "orthogonal" attention family, fully replacing Q/K/V
projections inside an otherwise-standard ViT (not an add-on, unlike QViT).
This project has built **two different circuit variants** of this idea, and
the report is careful to keep them separate:

- **RBS/Givens butterfly** (the variant reported in the CSV's headline id-23
  numbers): 1,536 trainable angles, **143,619** quantum params <
  174,851 sham < 207,619 classical multi-head attention. **Point-estimate
  quantum_wins on all 3 datasets tested**: Q−classical
  {0.000, +0.0019, +0.0005}, Q−sham {+0.001, +0.0030, +0.0075}.
- **RY+CNOT ring** (`train_qovt_rycnot.py` and its variants; this is the
  circuit §6's new ablation tests): a much smaller circuit, 48 trainable
  angles per U/V, **384** total attention-layer params vs sham's 32,768 and
  classical MHA's 65,536 — over two orders of magnitude leaner than the
  butterfly variant. Point-estimate result at Model_II N=500: quantum 0.9144
  vs sham 0.8812 (`EXPERIMENTS_MASTER.md`'s null-baseline table), the single
  largest point gap seen anywhere in the QOVT family, but also single-seed
  and untested until §6.

Both variants are **single seed, never statistically tested** in their
original runs — this is exactly the gap this project's next phase is
closing (§6, currently testing the RY+CNOT variant; the butterfly variant
is a candidate for the same treatment afterward). See §5 for why this
method, among all 32, was chosen
to carry the project forward.

### 3.7 The QVF-Hybrid campaign (ids 26, 27, 28, 29, 30, 31) — this project's largest single effort, and its self-correction

QVF-Hybrid (CNN(93k) ⊕ residual quantum branch, 143,655 params) was the
project's "final locked architecture" as of 2026-07-21, backed by a
three-tier claim chain and a multi-day, pre-registered significance
campaign. Two tiers still stand:

- **Vs. plain CNN**: floor-by-construction (zeroing the quantum branch
  reduces exactly to the plain classical model) — never a claimed advantage,
  correctly labeled "no model-level advantage claim."
- **Vs. the MAE-ViT paradigm** (id 30): a genuine, pre-registered, 3-seed
  ≥3% win in 6/7 cells (+2.4pp to +30.9pp), 1/19th the parameters, zero
  pretraining. **But this win belongs to the CNN's inductive bias, not the
  quantum branch** — a zero-quantum plain-linear CNN wins by a comparable
  margin over the same MAE-ViT baseline. Read correctly as "CNN family beats
  ViT+MAE paradigm at low label counts," not as quantum evidence.

The third tier — a **module-level quantum-vs-sham effect reported as
"statistically significant"** (p=0.0032, n=20, Holm-corrected, at Model_II
N=250) — is **retracted** (id 29). §4.2 covers why in detail; in short, an
independent audit found the sham control was itself broken (never reduced
to the true classical floor), the p-value fails multiple-comparisons
correction by roughly 1–2 orders of magnitude once the full search that
found this cell is accounted for, and the genuinely-new confirmatory seeds
alone are null. Related sub-experiments (QVF-WIDE, id 27; QVF-TDA, id 28;
the entanglement ablation, id 26; CommonTest generalization, id 31) are
individually noted in the CSV — the entanglement ablation's *internal*
ordering (ENT > NOENT > SHAM, 6/6 cells) is not itself invalidated by the
audit (it doesn't depend on the disputed p-value), but sits inside the same
now-abandoned architecture family.

### 3.8 A dataset-difficulty diagnostic, not a quantum result (id 25)

Model V (Michael Toomey 2024) is a harder, imbalanced, low-SNR dataset
(`axion`=17,100 / `cdm`=5,378 / `no_sub`=23,400). Every architecture tried
on it — ViT scratch/sham/quantum-head, QCNN sham/quantum-patches, QVF-angle
quantum/sham/classical, MAE fine-tune — lands at AUC≈0.50 regardless of
quantum use. **Verdict: inconclusive**, and explicitly not an architecture
problem: logistic regression on raw pixels is equally at chance, while an
overfit test on 300 samples reaches 0.90 *train* AUC, confirming the signal
exists but is drowned by within-class brightness variation (5–10× lower
pixel-level SNR than Models I–III) that no architecture tested — classical
or quantum — can see through without physics preprocessing (lens-model
subtraction) first. Deprioritized, not pursued further.

### 3.9 A different task entirely — QELP quantum-sensor metrology (id 32)

Not an image-classification result. Reproduces a theorem-backed,
literature-established quantum-memory advantage (CHSH/CCHL-type separation)
for estimating properties of quantum-sensor measurement data — up to 107×
error reduction with quantum memory vs. memoryless strategies at n=8, and an
assumption-free-detection advantage that is qualitatively different (a
classical strategy either bets on an unknown coupling axis and silently
misses the signal when wrong, or pays a 2.5–3× sample cost for
assumption-free coverage). Connected to the same physics target
(ultralight dark matter, same mass window as Model V's axion) but not
comparable to, and never conflated with, any classification result above.

---

## 4. Cross-cutting findings

### 4.1 The null-baseline discovery

Partway through the project, the missing control was identified: every
prior "quantum > sham" result in the NAE-bottleneck family (QVF-scratch,
QVF-Hybrid, the entanglement ablation) had been measured against a sham
that shared the *same* 8-dimensional measurement bottleneck as the quantum
arm — a handicap, not a floor. A plain 93,763-param CNN with a direct
linear head (no bottleneck, fewer parameters than any quantum variant in
the project) was benchmarked against the whole family and found to match or
beat every one of them — QVF-Q, QOVT-RYCNOT, QOVT-Givens, and the 2.72M-
param MAE-ViT — at nearly every (dataset, N), including beating QVF-quantum
by +0.092 AUC at exactly the low-N cell (Model_II, N=500) where the
quantum-regularization story was claimed strongest.

This finding doesn't erase the earlier quantum-vs-sham results — they're
still real within their own scope (quantum genuinely beats its matched
sham) — but it changes what that scope means: **"beats its own sham" is not
the same claim as "is competitive with the best available classical
model,"** and every subsequent report in this project (including this one)
keeps those two claims explicitly separate.

### 4.2 The QVF-Hybrid audit and retraction

At the researcher's explicit request to stop trusting this project's own
"significant" results without re-checking them, an independent adversarial
audit re-derived the QVF-Hybrid significance claim from raw SLURM logs and
source code (not from the narrative summary). Five findings:

1. **Broken control.** `train_qvf_opt.py`'s sham arm ran the NAE bottleneck
   unconditionally, never reducing to the true classical floor the "floor
   achieved" claim required. Empirically, sham scored *below* the plain-CNN
   floor (−0.006, p=.885) — meaning roughly 42% of the reported "+1.4%
   quantum contribution" was the control's own handicap, not a quantum
   effect.
2. **Multiple-comparisons shortfall.** Roughly 280 quantum-vs-classical
   comparisons were made across this project's full history before the
   confirmatory campaign. The reported p=0.0032 fails a Bonferroni
   correction over that full search by roughly 90×, and by roughly 26× even
   restricted to the ~82-comparison QVF-only sub-family — it survives only
   the *local* 6-test correction pre-registered for the confirmatory
   campaign in isolation.
3. **Non-independent confirmatory sample.** Splitting the n=20 confirmatory
   sample into the original discovery seeds (already seen before the test
   was designed) and the genuinely-new seeds shows the new-seeds-only half
   is null (p=0.097).
4. **Pre-registration timing.** Partially verifiable (file timestamps
   predate the job), but stage-2's protocol was written 106 seconds *after*
   stage-1's results had finished computing — i.e., after seeing which cell
   looked best.
5. **Seed-shrinkage precedent.** The most comparable cell shrank 6.2× from
   single-seed to n=20; one architecture-matched cell shrank 10.9× from n=3
   to n=10 seed luck alone — establishing this project's own rule of thumb
   that a new single-seed positive result here should be discounted 6–11×
   before being treated as real.

No fabricated numbers were found — every headline figure reproduces exactly
from raw logs. The problem was a broken control and an inflated
significance claim built on genuinely-run experiments, not fraud. The
corrected verdict (`retracted`) and findings 1–3 above are the public
record in `docs/BENCHMARK_DATABASE.csv` row 29, `docs/BENCHMARK.md`, and
this project's `README.md` — publicly self-corrected in commits `fb48f0c`
and `8364fdb`. Findings 4 and 5 are recorded in this project's internal
working notes (`qvf-audit-findings-20260731`) but have not yet been folded
into the public row-29 caveat text — worth doing in a follow-up commit,
since they're part of the same audit and no less load-bearing than 1–3.

---

## 5. Why QOVT was chosen as the forward direction

Given the record above, four considerations pointed to QOVT specifically:

**1. It is the architecturally cleanest comparison left standing.** Most of
this project's positive results (QVF-scratch, QVF-Hybrid, the entanglement
ablation) live inside the NAE-bottleneck family that §4.1 showed loses to a
plain CNN outright — their "wins" are real only against an internal,
already-compromised sham. QOVT's comparison is structurally different: the
quantum arm, the sham (unconstrained linear), and the classical arm
(standard multi-head attention) are **three variants of the exact same ViT
backbone**, differing only in how the attention projections are computed.
There is no shared bottleneck to relitigate — the null-baseline critique
that sank the NAE family's internal comparisons does not apply to QOVT's
*internal* quantum-vs-sham-vs-classical structure (it does still apply if
QOVT is compared *across* architecture families to the plain-CNN floor,
which is why this project scopes ongoing QOVT analysis to same-architecture
comparisons — see the project's own memory note on this scoping decision).

**2. It sits in the part of the architecture space this project's own
"surrogate theorem" says quantum should help.** §3.2 established a
consistent pattern across several independent experiments: quantum helps as
a low-dimensional *readout* (ids 6, 7, 19), and hurts or ties as a
high-dimensional *feature extractor* (ids 5, 8, 20, 22). QOVT's attention
mechanism is structurally a readout-like operation (a bounded orthogonal
transform gating which patches attend to which), not a raw feature
extractor like quanvolution (id 20, which lost badly) — it is the closest
fit to the one regime this project has repeatedly found favorable.

**3. It has never been given the scrutiny that caught QVF-Hybrid's flaw —
which is a reason to test it properly, not a reason to trust it yet.** Every
QOVT number in the database (id 23) is single-seed, never formally tested,
and — per an internal inventory audit (2026-08-06) — was produced by a
script family with real methodological gaps: no held-out test set (all
prior QOVT scripts report `max(val_AUC)` over training, the same split used
for model selection), no parameter-matched classical control in most
variants, and an LR asymmetry favoring the quantum arm. In other words:
QOVT was chosen *before* being cleared of the same category of problem that
sank QVF-Hybrid, specifically so that problem could be found and fixed
first, rather than discovered after another "significant" claim shipped.
That is now underway — see §6.

**4. Fewest parameters of the three arms, with no parameter-matched control
anywhere in the reference paper either.** Cherrat et al. (2022,
arXiv:2209.08167) itself only compares its quantum architectures against
classical ViT/OrthoFNN/AutoML baselines, never a rank-matched linear
control — the same gap this project is now filling with a genuine
low-rank-matched "matched" arm (§6), going further than the reference paper
did.

QOVT is not being presented as a replacement headline claim. It is the
project's best remaining candidate for a real, defensible result — chosen
precisely because its comparison structure and its literature (§3.6's
reference paper) give it the cleanest path to either confirm or kill a
quantum-attention claim on this task, following the same standard the
QVF-Hybrid retraction was held to.

---

## 6. Pre-registered QOVT ablation — first results

Before running any new ablations, the existing 8-script QOVT family was
audited the same way QVF-Hybrid's code was. Findings (full detail in the
project's working notes): all 8 scripts have real completed runs (no
fabrication), but methodological gaps — no held-out test set, missing
parameter-matched controls, and an LR asymmetry — mean none of the existing
numbers can yet support a claim, "wins" included.

**Response:** a new script,
[`experiments/08_qovt_ablation/train_qovt_ablation.py`](../experiments/08_qovt_ablation/train_qovt_ablation.py),
was written to fix these gaps, and a pre-registration document,
[`docs/QOVT_ABLATION_PREREGISTRATION.md`](QOVT_ABLATION_PREREGISTRATION.md),
was committed **before any run happened**, fixing the test cells (quantum
vs. a new rank-1 low-rank "matched" control, and vs. the existing sham, on
Model_II and Model_III at N=500), the seed count (10, seeds 42–51), the
statistical test (paired one-sided Wilcoxon, Holm-corrected across 4
primary tests, α=0.01), and the practical-significance bar (≥2% AUC margin,
consistent direction in ≥8/10 seeds, on top of the p-value) — specifically
so this round cannot repeat the QVF-Hybrid campaign's unverifiable-timing
and discovery/confirmatory-overlap problems.

**Self-caught bug in the new script, disclosed before any result was
trusted:** the first 80-run sweep (job 237915) produced an implausible
+23pp to +35pp AUC gap, quantum beating every control on all 4 primary
tests. Rather than reporting this, the training curves were inspected
directly — `matched` never moved off chance for all 30 epochs, `sham` rose
then plateaued near chance — and the cause was found: the new script's own
attempted fix for the *original* scripts' LR asymmetry had applied the
quantum-circuit-calibrated learning rate (1e-2) uniformly to every arm's
attention parameters, which destabilized the classical arms' much larger
linear layers instead of leveling the playing field. This was disclosed and
fixed (single uniform learning rate is now the script's default; the raw
invalidated run is preserved, not deleted, at
`experiments/08_qovt_ablation/results_ablation_RUN1_INVALIDATED_qlr_bug.jsonl`)
before the corrected rerun (job 251067) was submitted.

**Results (job 251067, 80/80 runs, all 4 primary tests reported regardless
of outcome, exactly as pre-registered):**

| # | Test | Δ (AUC) | pos/10 | Holm p | Verdict |
|---|---|---|---|---|---|
| 1 | Q > matched, Model_II | +0.3366 | 10/10 | 0.0039 | **CANDIDATE FINDING** |
| 2 | Q > matched, Model_III | +0.3227 | 10/10 | 0.0039 | **CANDIDATE FINDING** |
| 3 | Q > sham, Model_II | +0.1566 | 10/10 | 0.0039 | **CANDIDATE FINDING** |
| 4 | Q > sham, Model_III | +0.0739 | 7/10 | 0.0137 | directional only, n.s. |

Three of four primary tests clear the pre-registered bar (Holm p<0.01 AND
|Δ|≥2% AND ≥8/10 seeds) — **this is the first QOVT result in this project's
history to survive a multi-seed, pre-registered statistical test.** The two
"matched" wins (rank-1 low-rank control, tests 1–2) are the least surprising
— a rank-1 linear map is a weak function class, and this control was
explicitly disclosed as not an exact parameter match. The **"sham"
comparison is the more informative test** (32,768-param unconstrained
Linear, a materially stronger control) and is genuinely mixed: significant
on Model_II, not on Model_III (test 4 fails both the corrected p-value and
the seed-consistency bar). This is reported as a real negative result for
that cell, not rounded up.

**Held to this project's own standard, this is a strong candidate, not yet
a confirmed finding.** The three tests that cleared the pre-registered bar
(15.7–33.7pp) are an order of magnitude larger than the QVF-Hybrid effect
that shrank 6–11× under replication and ultimately failed — reassuring, but
not a substitute for actually replicating it, and it does not rescue test 4
(+7.4pp, did not clear the bar). The recommended next step, before treating
this as
settled, is an independent confirmatory round at a second N with a fresh
seed range (e.g. seeds 60–69) — the discipline the QVF-Hybrid campaign got
right in its pre-registration design and wrong in its execution (its
"confirmatory" seeds partly overlapped the discovery sample). Full numbers,
per-arm means, and the complete reasoning are in
[`docs/QOVT_ABLATION_PREREGISTRATION.md`](QOVT_ABLATION_PREREGISTRATION.md)'s
Run 2 section; the result is also recorded as
[`docs/BENCHMARK_DATABASE.csv`](BENCHMARK_DATABASE.csv) row 33, with a
verdict deliberately labeled `quantum_wins (directional, single
pre-registered round, not yet independently replicated)` rather than a bare
`quantum_wins`, so its actual evidentiary weight stays visible at a glance
rather than reading as more settled than it is.

**Important scope note:** this ablation tests only the **RY+CNOT ring**
circuit (384 attention-layer params) — a different circuit from the
**RBS/Givens butterfly** variant reported in row 23 (1,536 attention-layer
params, **4×** RY+CNOT's, despite both landing at nearly the same *total*
model size since the CNN backbone dominates the count). The two are not
interchangeable; a result on one says nothing directly about the other, and
both would need the same pre-registered treatment before either could be
called established (see §3.6).

**Planned next steps**, to complete the reference-paper-equivalent analysis
this project set out to do:
- **Circuit-topology ablation** (the reference paper's Table 1): Pyramid,
  X, and Butterfly connectivity patterns for the orthogonal attention layer
  — currently only the RY+CNOT ring variant has been tested.
- **Architecture-type framing** (the reference paper's Table 2): explicitly
  categorizing this project's QOVT variants against the reference paper's
  A/B/C/D taxonomy (patch-wise / orthogonal-transformer / direct-quantum-
  attention / compound-transformer) to identify which regime, if any, this
  project's implementation occupies and whether the untested regimes are
  worth building.
- **Multi-seed extension to more (dataset, N) cells**, once the N=500
  pre-registered result is in, following the same discipline (write the
  next test down before running it, not after).

---

## 7. Standing lessons for this project going forward

1. **A single favorable run is not a result.** Every genuine positive
   finding in this project's history that survived scrutiny (QVF-scratch's
   21/22-point sweep, the entanglement ablation's 6/6 strict ordering) did
   so because it was checked across many points, not because one number
   looked good. Every finding that didn't survive (QAE anomaly, QVF-Hybrid's
   significance claim) failed exactly where it hadn't been checked that way
   yet.
2. **The control has to earn the label "sham."** Two of this project's worst
   mistakes (the NAE-bottleneck sham, and `train_qvf_opt.py`'s broken sham)
   were both a sham that didn't actually reduce to a fair, unhandicapped
   classical baseline. Every new ablation script should be checked for this
   specifically, the same way `train_qovt_ablation.py`'s own LR bug was
   caught this round.
3. **Discount single-seed effects 6–11× before trusting them**, per this
   project's own measured seed-shrinkage pattern — a number worth applying
   to every future QOVT cell before it's reported as a finding.
4. **Retraction is not failure — it's the methodology working.** The
   QVF-Hybrid retraction and this round's LR-bug catch are both instances of
   the same discipline: check your own best-looking number harder than
   anyone else will, before anyone else has to.

---

## References

- Full method database: [`docs/BENCHMARK_DATABASE.csv`](BENCHMARK_DATABASE.csv),
  [`docs/BENCHMARK.md`](BENCHMARK.md)
- Full narrative lab log: [`docs/EXPERIMENTS_MASTER.md`](EXPERIMENTS_MASTER.md)
- QOVT ablation pre-registration and results:
  [`docs/QOVT_ABLATION_PREREGISTRATION.md`](QOVT_ABLATION_PREREGISTRATION.md)
- Reproducible code for the retracted QVF-Hybrid claim, bug disclosed inline:
  [`experiments/07_qvf_final_locked/`](../experiments/07_qvf_final_locked/)
- QOVT ablation code: [`experiments/08_qovt_ablation/`](../experiments/08_qovt_ablation/)
- Reference paper for QOVT: Cherrat, Kerenidis, Mathur, Landman, Strahm, Li,
  "Quantum Vision Transformers," *Quantum* **8**, 1265 (2024),
  [arXiv:2209.08167](https://arxiv.org/abs/2209.08167)
- Original project proposal (historical, superseded — see its own status
  note): [`PROPOSAL.md`](../PROPOSAL.md)
