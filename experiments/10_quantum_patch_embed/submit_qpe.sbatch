#!/bin/bash
#SBATCH -J qpe_sweep
#SBATCH -A gov114009
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:30:00
#SBATCH -o /home/leo07010/gsoc2026-deeplense-qml/experiments/10_quantum_patch_embed/logs/sbatch-%j.out
#SBATCH -e /home/leo07010/gsoc2026-deeplense-qml/experiments/10_quantum_patch_embed/logs/sbatch-%j.err
set -euo pipefail

# NOTE (2026-08-14): the qc_llm conda env lost numpy/torch/scipy on
# 2026-08-13 (partial wipe, /home at 97% disk) -- system python has the
# full working stack (torch 2.8.0+cu128, CUDA OK), so no conda here.
module purge
module load cuda/13.0
export QPE_DEPTH=4

REPO=/home/leo07010/gsoc2026-deeplense-qml/experiments/10_quantum_patch_embed
cd "$REPO"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION dataset=${1:?}"
echo "[SLURM] python=$(which python3)"
python3 -c "import torch, numpy, sklearn, scipy; \
  assert torch.cuda.is_available(), 'no CUDA'; \
  print('env ok torch', torch.__version__)" || { echo '[FATAL] env broken'; exit 1; }
nvidia-smi -L || true

bash run_qpe_sweep.sh "$1"
