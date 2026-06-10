#!/bin/bash
# QAE ensemble FULL run: 4 arms, single seed 42 (per user request — no multi-seed).
# Assumes ssl_features_I.npz already extracted by the smoke job.
set -euo pipefail
cd /home/leo07010/mae-lensing

for arm in maha matched sham quantum; do
    echo "════════ FULL arm=$arm seed=42 ════════"
    python -u train_qae_ensemble.py --arm "$arm" --seeds 42 \
           --out_csv qae_ensemble_results.csv
done
echo "[FULL] all arms done"
