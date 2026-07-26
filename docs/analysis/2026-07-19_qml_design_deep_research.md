# Deep research: can any QML design clear the ≥3% bar on DeepLense? (2026-07-19)

Method: 5-angle web fan-out → 15 sources fetched → 3-vote adversarial verification
per claim (2/3 refutes kills). All findings below survived 3-0 unless noted.
Run: deep-research workflow, 106 agents.

## 1. The "Quantum Topological Data Encoding" paper

Best match: **arXiv:2605.28927 — "Quantum encodings that preserve persistent
homology" (Parzygnat & Vlasic, May 2026)**. Pure admissibility-theory paper:
which quantum encodings preserve persistent-homology invariants of the data.
Self-declared "first step" — **no classifier, no datasets, no experiments, no
qubit/parameter counts**. Nothing transferable to a lensing pipeline; its only
actionable content is that the mechanism sits at the data-encoding stage.

Related line (same authors): arXiv:2209.10596 (QIC 2023) shows angle/amplitude/
IQP encodings each DISTORT data topology (one 2k-point synthetic dataset, no
classifier); arXiv:2412.17772 (PRR 2025) is category-theory perspective, zero
empirics. The line never shows topology preservation improves classification.

⚠ Identity caveat: no paper titled exactly "Quantum Topological Data Encoding"
exists; if Leo means a different paper, re-verify before citing.

## 2. Quantum TDA is ruled out as a design family

- Only provable exponential speedup (arXiv:2410.21258, PRX Quantum 2026) is a
  DECISION problem (does a given hole persist), not barcode/Betti feature
  extraction for ML.
- McArdle–Gilyén–Berta (Quantum 10, 2058, 2026): "currently no evidence" of
  exponential speedup for practical TDA; quantum-inspired classical power
  method within a quadratic factor (near-dequantization).
- Classical persistent homology on 64×64 images is computationally trivial —
  there is no runtime bottleneck for quantum to relieve.

## 3. Field-level verdict on QML image classification (as of 2026-03)

Feb 2026 comprehensive review (arXiv:2603.06644): "no study has established a
scalable and reproducible quantum advantage over state-of-the-art classical
vision baselines under an explicit, budget-matched resource contract";
empirical wins are "more often explained by baseline choice, inductive-bias
effects, and hybridization." Corroborated by Bowles/Ahmed/Schuld 160-dataset
benchmark (arXiv:2403.07059).

Best published "wins" collapse under our fairness bar:
- Senokosov et al. (arXiv:2304.09224, MLST 2024): 8× parameter-efficiency uses
  a DERIVED baseline (frozen CNN trunk, PQC↔dense swap), not an independently
  tuned compact CNN; quanvolution result is a statistical tie (0.67±0.01 vs
  0.66±0.02 on 500 MNIST images).
- arXiv:2402.10540: hybrid architectures compared only against each other, no
  classical baseline at all.
- Only defensible claim-shape in the literature: **parameter efficiency at
  matched accuracy** — and even that rests on non-optimized comparators.

## 4. DeepLense prior art

All ~14-16 DeepLense GSoC projects (2021-2024) are classical (ViT, Lensiformer,
PINN, equivariant NN, SSL); "quantum" appears nowhere in the repo. ML4SCI's
quantum work is under QMLHEP (LHC), not lensing. **Our quantum-vs-93k-CNN
comparison is effectively the first controlled DeepLense QML benchmark** —
a positioning asset for the proposal regardless of sign.

## 5. The one mechanism-grounded positive signal

The axion class is EXPLICITLY defined by topological features: "vortex
substructures refer to specific topological features" (DeepLense README);
Alexander et al. (ApJ, arXiv:1909.07346): line-like vortex defects vs
point-mass CDM subhalos vs smooth halos. This licenses **classical persistent
homology** (sublevel+superlevel filtrations → persistence images / Betti
curves) as a hypothesis-driven, tiny-parameter feature channel — potentially
adding information the strided-conv CNN discards, especially at low N.
Caveat: field-theoretic defects ≠ image-level PH; testable hypothesis, not an
expected win. No published TDA-on-lensing benchmark exists.

## 6. Recommended design (single architecture, classical-first)

Keep the 93,763-param plain CNN; add a parallel TDA channel:
persistence images (2 homology dims × 16×16) from sublevel+superlevel
filtrations → small MLP (~1-3k params) → concat before the linear head.
Total budget < ~97k.

Controls: (1) sham channel = same-size MLP on parameter-matched
non-topological input (shuffled persistence images or 16×16 downsampled
pixels); (2) plain-CNN null; (3) ≥3 seeds; (4) low-N sweep N∈{500,1000,1500}
(at full data 0.9986 nothing can show ≥3%).

Quantum appendix ONLY if the classical TDA channel first shows a gain:
topology-preserving encoding (2605.28927 admissibility criteria) of the PH
feature vector into a small PQC head vs parameter-matched classical head on
identical features — framed strictly as parameter-efficiency at matched
accuracy. This escapes all prior failure modes: new information enters as
classical features (no measurement bottleneck, no raw-pixel amplitude
encoding); quantum, if any, is a readout on features already proven
informative.

**Honest probabilities**: quantum component clearing ≥3%: **<10%** (zero
literature precedent under fair controls); classical TDA channel clearing ≥3%
at N=500/class: **~20-35%** (real mechanism, no precedent either way); if both
null → well-controlled negative result + parameter-efficiency framing.

## 7. Open questions / cheap next steps

1. **CPU-only pilot (hours, no GPU)**: do persistence images on actual
   Model_II images separate axion vs CDM at N=500? TDA+MLP alone vs chance.
2. Un-researched angle: E(2)-rotation-equivariant compact classical CNN at
   low N — if it beats the 93k CNN, it becomes the new bar (verification
   produced no surviving claims either way on equivariant QML).
3. Post-2026-03 literature watch: any budget-matched QML win would revise <10%.
4. If TDA gains: instantiate an admissible topology-preserving encoding on
   ≤8 qubits vs sham (non-admissible) encoding — the only remaining route to
   a defensible quantum-specific claim.

Sources (key): arXiv:2605.28927, 2209.10596, 2412.17772, 2410.21258,
Quantum 10:2058 (2026), 2506.01432, 2603.06644 (Feb 2026 review), 2403.07059,
2304.09224, 2402.10540, github.com/ML4SCI/DeepLense, arXiv:1909.07346.
