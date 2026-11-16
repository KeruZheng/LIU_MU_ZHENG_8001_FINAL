#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROBOTIC_ROOT=${1:?Absolute robotic project root required}
[[ "$ROBOTIC_ROOT" == /* && "${ROBOTIC_ROOT##*/}" == robotic ]]
cd "$ROBOTIC_ROOT"
mkdir -p results logs/slurm
export PYTHONPATH="$ROBOTIC_ROOT/src"
export HF_HOME="$ROBOTIC_ROOT/envs/huggingface"
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
sha256sum --check manifests/code.sha256 manifests/data.sha256
envs/robotic/bin/python scripts/preflight.py --require-cuda --full-model \
  --output "results/preflight_${SLURM_JOB_ID}.json"
envs/robotic/bin/python -m unittest discover -s tests -v
