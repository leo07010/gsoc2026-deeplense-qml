#!/bin/bash
#SBATCH -J qovt_ablation
#SBATCH -A gov114009
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /home/leo07010/gsoc2026-deeplense-qml/experiments/08_qovt_ablation/logs/sbatch-%j.out
#SBATCH -e /home/leo07010/gsoc2026-deeplense-qml/experiments/08_qovt_ablation/logs/sbatch-%j.err
set -euo pipefail

module purge
module load miniconda3/26.1.1 cuda/13.0
export PATH="$HOME/.conda/envs/qc_llm/bin:$PATH"

REPO=/home/leo07010/gsoc2026-deeplense-qml/experiments/08_qovt_ablation
cd "$REPO"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION"
echo "[SLURM] python=$(which python3)"
nvidia-smi -L || true

bash run_ablation_sweep.sh
