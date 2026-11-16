#!/bin/bash
set -euo pipefail
ROBOTIC_ROOT=${1:?Absolute robotic project directory required}
ROBOTIC_BASE_PYTHON=${2:-python3.10}
[[ "$ROBOTIC_ROOT" == /* && "${ROBOTIC_ROOT##*/}" == robotic ]]
cd "$ROBOTIC_ROOT"
"$ROBOTIC_BASE_PYTHON" -m venv envs/robotic
export PIP_CACHE_DIR="$ROBOTIC_ROOT/envs/pip-cache"
envs/robotic/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
envs/robotic/bin/python -m pip install -r configs/requirements.txt
envs/robotic/bin/python -m pip freeze > manifests/cluster_environment.txt
# Run CUDA and model checks only through Slurm after installation.
