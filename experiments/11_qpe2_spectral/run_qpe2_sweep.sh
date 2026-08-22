#!/bin/bash
# QPE-2 sweep: 6 embed arms x 20 seeds x N in {100, 250}, ONE dataset per
# invocation (model_II or model_III as $1). 240 runs per job, 4-way GPU
# concurrency (same pattern as QPE-1, which did 80 runs in ~23 min).
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR=/home/leo07010/mae-lensing
DATASET=${1:?usage: run_qpe2_sweep.sh <model_II|model_III>}
SEEDS=$(seq 42 61)
ARMS="quantum skew48 butterfly cayley dctfix conv"
NS="100 250"
OUT="results_qpe2_${DATASET}.jsonl"
FAILED="failed_runs_${DATASET}.txt"
MAXJOBS=4
mkdir -p logs
# truncate BOTH: appending to a stale $OUT on resubmission would silently
# duplicate seeds and break the paired Wilcoxon
rm -f "$FAILED" "$OUT"

run_one () {
    local embed=$1 seed=$2 n=$3
    python3 -u train_qpe2.py \
        --data "${DATA_DIR}/${DATASET}.npz" \
        --embed "$embed" --seed "$seed" --n_per_class "$n" \
        --epochs 96 --batch_size 32 --out_json "$OUT" \
        > "logs/${embed}_${DATASET}_N${n}_seed${seed}.out" 2>&1
}

n_launched=0
for n in $NS; do
    for embed in $ARMS; do
        for seed in $SEEDS; do
            while (( $(jobs -rp | wc -l) >= MAXJOBS )); do
                wait -n || true
            done
            (
                if run_one "$embed" "$seed" "$n"; then
                    echo "[$(date +%H:%M:%S)] done: ${embed} ${DATASET} N${n} seed${seed}"
                else
                    echo "[FAILED] ${embed} ${DATASET} N${n} seed${seed}"
                    echo "${embed} ${n} ${seed}" >> "$FAILED"
                fi
            ) &
            n_launched=$((n_launched+1))
        done
    done
done
wait
echo "[SWEEP DONE] ${DATASET}: ${n_launched} runs launched, results in ${OUT}"
if [[ -s "$FAILED" ]]; then
    echo "[WARNING] $(wc -l < "$FAILED") failed runs listed in ${FAILED}"
fi
