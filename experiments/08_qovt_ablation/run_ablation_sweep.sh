#!/bin/bash
# Driver for the pre-registered QOVT ablation (docs/QOVT_ABLATION_PREREGISTRATION.md).
# 4 arms x 2 datasets x 10 seeds x N=500 = 80 runs. Model is tiny (<=210k params);
# 4-way background parallelism on a single GPU is safe (no memory contention) and
# cuts wall time from ~2.4h serial to well under an hour.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=/home/leo07010/mae-lensing
OUT=results_ablation.jsonl
: > "$OUT"   # fresh file -- this is a first run, not a resume

MODES="quantum matched sham classical"
DATASETS="model_II model_III"
SEEDS="42 43 44 45 46 47 48 49 50 51"
MAXJOBS=4

run_one() {
    local mode=$1 data=$2 seed=$3
    python3 -u train_qovt_ablation.py \
        --data "${DATA_DIR}/${data}.npz" \
        --mode "$mode" --n_per_class 500 --epochs 30 --seed "$seed" \
        --out_json "$OUT" \
        > "logs/${mode}_${data}_seed${seed}.out" 2>&1
}
mkdir -p logs

n=0
for data in $DATASETS; do
  for mode in $MODES; do
    for seed in $SEEDS; do
      run_one "$mode" "$data" "$seed" &
      n=$((n+1))
      if (( n % MAXJOBS == 0 )); then wait; fi
    done
  done
done
wait
echo "[SWEEP DONE] $(wc -l < "$OUT") results written to $OUT"
