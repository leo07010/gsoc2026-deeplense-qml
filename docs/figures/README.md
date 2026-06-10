# Method figures (TikZ)

Compiled flowcharts of the full pipeline. Sources are `*.tex`; rebuild with
`pdflatex <name>.tex && pdftoppm -png -r 150 <name>.pdf <name>`.

## Classical baseline
![Classical MAE](classical_mae.png)

## Quantum methods
| Method | Figure |
|---|---|
| QCT-scratch (Quantum-Classical Transformer) | ![QCT](qct.png) |
| QVF-scratch (Neural Amplitude Encoding) | ![QVF](qvf.png) |
| Dual-Encoder + FiLM fusion head | ![Dual](dualenc.png) |

## Shared components
| Component | Figure |
|---|---|
| Combining (fusion) heads: concat / gated / FiLM | ![Fusion](fusion_heads.png) |
| Classification head & readout | ![Classifier](classifier_head.png) |
