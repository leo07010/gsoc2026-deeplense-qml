#!/bin/bash
# Round 2: butterfly-circuit quantum arm only (20 runs). matched/sham/classical
# are reused from results_ablation.jsonl (job 251067) -- see
# docs/QOVT_ABLATION_PREREGISTRATION.md "Round 2" for why that's valid.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=/home/leo07010/mae-lensing
OUT=results_butterfly.jsonl
: > "$OUT"

DATASETS="model_II model_III"
SEEDS="42 43 44 45 46 47 48 49 50 51"
MAXJOBS=4

run_one() {
    local data=$1 seed=$2
    python3 -u train_qovt_ablation.py \
        --data "${DATA_DIR}/${data}.npz" \
        --mode quantum --circuit butterfly --n_per_class 500 --epochs 30 --seed "$seed" \
        --out_json "$OUT" \
        > "logs/butterfly_${data}_seed${seed}.out" 2>&1
}
mkdir -p logs

n=0
for data in $DATASETS; do
  for seed in $SEEDS; do
    run_one "$data" "$seed" &
    n=$((n+1))
    if (( n % MAXJOBS == 0 )); then wait; fi
  done
done
wait
echo "[SWEEP DONE] $(wc -l < "$OUT") results written to $OUT"
