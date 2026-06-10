#!/bin/bash
# QAE ensemble pipeline smoke: extract leakage-free SSL features, then run
# all 4 arms in smoke mode (1 seed, 3 epochs, 256 fit samples per class).
set -euo pipefail
cd /home/leo07010/mae-lensing

python -u extract_features_ssl.py --data model_I.npz --encoder enc_I.pth \
       --out ssl_features_I.npz

for arm in maha matched sham quantum; do
    echo "════════ SMOKE arm=$arm ════════"
    python -u train_qae_ensemble.py --arm "$arm" --seeds 42 --smoke \
           --out_csv qae_smoke_results.csv
done
echo "[SMOKE] all arms done"
