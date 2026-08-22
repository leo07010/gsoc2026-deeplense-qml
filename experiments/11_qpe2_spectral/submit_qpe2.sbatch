#!/bin/bash
#SBATCH -J qpe2_sweep
#SBATCH -A gov114009
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:50:00
#SBATCH -o /home/leo07010/gsoc2026-deeplense-qml/experiments/11_qpe2_spectral/logs/sbatch-%j.out
#SBATCH -e /home/leo07010/gsoc2026-deeplense-qml/experiments/11_qpe2_spectral/logs/sbatch-%j.err
set -euo pipefail

# qc_llm conda env is broken (numpy/torch wiped 2026-08-13); system python
# has the full stack -- see memory qc-llm-env-broken.
module purge
module load cuda/13.0
export QPE2_DEPTH=4

REPO=/home/leo07010/gsoc2026-deeplense-qml/experiments/11_qpe2_spectral
cd "$REPO"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION dataset=${1:?}"
echo "[SLURM] python=$(which python3)"
python3 -c "import torch, numpy, sklearn, scipy; \
  assert torch.cuda.is_available(), 'no CUDA'; \
  print('env ok torch', torch.__version__)" || { echo '[FATAL] env broken'; exit 1; }
nvidia-smi -L || true

bash run_qpe2_sweep.sh "$1"
