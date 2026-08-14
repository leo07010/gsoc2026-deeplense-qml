#!/bin/bash
# QPE sweep: 4 embed arms x 20 seeds x N=500, ONE dataset per invocation
# (pass model_II or model_III as $1). 20 seeds (not 10): power analysis on
# the 08 ablation's observed paired-diff SDs showed n=10 detects a true
# 5pp effect only ~28% of the time at the Holm bar; n=20 raises that to
# ~85%. 4-way GPU concurrency (08's pattern, job 251310 anchor) keeps the
# wall time inside the dev window anyway.
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR=/home/leo07010/mae-lensing
DATASET=${1:?usage: run_qpe_sweep.sh <model_II|model_III>}
SEEDS=$(seq 42 61)
ARMS="quantum scramble dct conv"
OUT="results_qpe_${DATASET}.jsonl"
FAILED="failed_runs_${DATASET}.txt"
MAXJOBS=4
mkdir -p logs
rm -f "$FAILED"

run_one () {
    local embed=$1 seed=$2
    python3 -u train_qpe.py \
        --data "${DATA_DIR}/${DATASET}.npz" \
        --embed "$embed" --seed "$seed" --n_per_class 500 \
        --epochs 48 --out_json "$OUT" \
        > "logs/${embed}_${DATASET}_seed${seed}.out" 2>&1
}

n=0
for embed in $ARMS; do
    for seed in $SEEDS; do
        while (( $(jobs -rp | wc -l) >= MAXJOBS )); do
            wait -n || true
        done
        (
            if run_one "$embed" "$seed"; then
                echo "[$(date +%H:%M:%S)] done: ${embed} ${DATASET} seed${seed}"
            else
                echo "[FAILED] ${embed} ${DATASET} seed${seed}"
                echo "${embed} ${seed}" >> "$FAILED"
            fi
        ) &
        n=$((n+1))
    done
done
wait
echo "[SWEEP DONE] ${DATASET}: ${n} runs launched, results in ${OUT}"
if [[ -s "$FAILED" ]]; then
    echo "[WARNING] $(wc -l < "$FAILED") failed runs listed in ${FAILED}"
fi
