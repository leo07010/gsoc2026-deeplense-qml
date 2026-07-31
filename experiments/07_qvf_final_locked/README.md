# 07_qvf_final_locked

Scripts ported from the private HPC working copy (`mae-lensing`) that produced the
QVF (Quantum-enhanced Vision... amplitude-encoding) family of results in
[`../../docs/BENCHMARK_DATABASE.csv`](../../docs/BENCHMARK_DATABASE.csv). These were
previously undocumented in this public repo — the CSV cited them as
"not present in this repo checkout". Ported verbatim (comments/logic unchanged)
except genericizing any machine-specific paths; none were found hardcoded in
these 7 files.

## Scripts

| script | what it does | CSV row(s) (`id`) |
|---|---|---|
| `qvf_plain_baseline.py` | Plain CNN, no quantum branch at all (linear or MLP head). The null baseline that Pareto-dominates the whole quantum family. | `1` |
| `train_qvf_scratch.py` | CNN → neural amplitude encoding (NAE) → 8/10/12-qubit `AmplitudeEmbedding` + `StronglyEntanglingLayers` → ⟨Z⟩ → head, trained end-to-end. `--sham` swaps the circuit for `Linear(256→K)`. Imports `quantum_qvf.py`. | `6` (amplitude encoding), `7` (angle-encoding mode, same file) |
| `quantum_qvf.py` | Shared circuit/NAE module (`_circuit`, `NeuralAmplitudeEncoding`, `QVFClassifier`, N_Q/DIM/K_LATENT constants). **Imported by `train_qvf_scratch.py`** (`from quantum_qvf import _circuit, enc_shape, K_LATENT, N_Q, DIM`) — required dependency for that script. Not imported by any of the other 5 scripts here (they each define their circuit inline). | supports `6`/`7` |
| `train_qvf_hybrid.py` | "Locked" architecture: `h=CNN(x)` (128-dim) run in parallel with `m=wide-readout(PQC(NAE(h)))` (52-dim); `logits=Linear([LN(h);LN(m)])`. Zeroing `m` reduces exactly to the plain-CNN floor by construction. `--sham` swaps the circuit for `Linear(256→52)+Tanh`. | `29` |
| `train_qvf_opt.py` | Same architecture as `train_qvf_hybrid.py` but with `--nq {8,10,12}` and `--qlr` (separate LR for circuit angles) as CLI levers. **Contains the disclosed sham-control bug** — see below. | `29` |
| `train_qvf_hybnoent.py` | Entanglement-ablation arm of the hybrid architecture: same `h⊕m` residual design, but the circuit's `StronglyEntanglingLayers` (with CNOT ring) is replaced by per-qubit `qml.Rot` only (same 96 angles, no entanglement). | `26` (best match by architecture — see caveat below) |
| `train_qvf_noent.py` | Entanglement-ablation arm of the **scratch** architecture (matches `train_qvf_scratch.py`'s single 8-dim ⟨Z⟩ readout, no hybrid residual branch), same no-CNOT substitution. | `26`, also cited in `29`'s code_file list |

**Caveat on `26`/`29` script attribution**: `BENCHMARK_DATABASE.csv` cites only
`train_qvf_noent.py` for both row `26` ("Same architecture ... as QVF-Hybrid") and
row `29`. By actual code structure, `train_qvf_hybnoent.py` is the closer match to
"QVF-Hybrid architecture, entanglement ablated" (same 128+52 residual design,
`K_WIDE`=52 readout), while `train_qvf_noent.py` ablates the older *scratch*
architecture (8-dim readout, no residual branch). Both are ported here since both
were cited as distinct jobs in the experiment log; they are **not** duplicates —
they ablate two different base architectures, not the same one twice.

## Data

All scripts take `--data path/to/model_X.npz` (`X` ∈ `I`/`II`/`III`, or `dataset1.npz`
— note `dataset1 ≡ model_II` per project memory, they are byte-identical, don't run
both). These `.npz` caches are **not** the raw DeepLense images described in
[`../../data/README.md`](../../data/README.md) — they're produced from that raw data
by `../00_baselines/cache_model.py`:

```bash
python ../00_baselines/cache_model.py --root <path-to-raw-Dataset-class-dir> --out model_II.npz
```

## CLI invocations

Each script's flags are documented in its own `argparse` block (all support
`--data`, `--epochs`, `--seed`, `--n_per_class`, `--smoke`; quantum-family scripts
add `--sham`; `train_qvf_opt.py` adds `--nq` and `--qlr`). Example syntax:

```bash
python train_qvf_opt.py --data model_II.npz --nq 8 --n_per_class 250 --seed 42
python train_qvf_opt.py --data model_II.npz --nq 8 --n_per_class 250 --seed 42 --sham
```

**Note**: I grepped `../../docs/EXPERIMENTS_MASTER.md` for the exact historical
command lines used to produce the headline runs (e.g. the Stage-2 20-seed
Model_II/N=250 campaign behind row `29`) and could not find literal invocation
strings there — it documents results/tables, not shell commands. The syntax above
is derived from each script's own `argparse` definitions and the parameters
(dataset, N, seed range) stated in the CSV/EXPERIMENTS_MASTER.md text, not a
verified historical transcript. Do not treat it as a copy-paste of what was
actually run.

## Known bug — read before reusing `train_qvf_opt.py`

`train_qvf_opt.py` contains a disclosed-but-not-fixed bug in `QVFOpt.forward()`
(see the comment block directly above it in the source): the NAE bottleneck runs
unconditionally before the sham/quantum branch, so the sham control never reduces
to the true classical floor. This inflates the reported "+1.4% quantum
contribution" claim. Full corrected verdict, statistical audit, and numbers:
[`../../docs/BENCHMARK_DATABASE.csv`](../../docs/BENCHMARK_DATABASE.csv) row `id=29`
("QVF-Hybrid / QVF-OPT / Significance Campaign ... [CORRECTED VERDICT]").
The bug is preserved, not fixed, in this port — this file is kept exactly as it
produced the disputed result.
