# Method figures

TikZ flowcharts of the three live quantum methods (compiled from `*.tex`).

| Method | Figure |
|---|---|
| QCT-scratch (Quantum-Classical Transformer) | ![QCT](qct.png) |
| QVF-scratch (Neural Amplitude Encoding) | ![QVF](qvf.png) |
| Dual-Encoder + FiLM fusion head | ![Dual](dualenc.png) |

Rebuild: `pdflatex <name>.tex && pdftoppm -png -r 150 <name>.pdf <name>`
