#!/bin/bash
# GPU job runner for the upgraded cluster (H200, Lmod refresh, qc_llm env).
# Usage:  sbatch -J name -t 00:30:00 --gres=gpu:1 slurm/run2.sbatch <cmd...>
#SBATCH -A mst114318
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /home/leo07010/mae-lensing/slurm/logs/%x-%j.out
#SBATCH -e /home/leo07010/mae-lensing/slurm/logs/%x-%j.err
set -euo pipefail

module purge
module load miniconda3/26.1.1 cuda/13.0
export PATH="$HOME/.conda/envs/qc_llm/bin:$PATH"

REPO=/home/leo07010/mae-lensing
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION"
echo "[SLURM] python=$(which python)"
nvidia-smi -L || true
echo "[SLURM] cmd: $*"
echo "============================================================"
srun "$@"
