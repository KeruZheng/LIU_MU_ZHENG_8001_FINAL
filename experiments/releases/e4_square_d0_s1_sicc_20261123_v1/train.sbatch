#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
set -euo pipefail
umask 0002
E4_PROJECT=${1:?project root required}
E4_RELEASE=${2:?release name required}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
export XDG_CACHE_HOME="$E4_PROJECT/operations/e4_parallel_20261123/cache/$SLURM_JOB_ID"
mkdir -p "$XDG_CACHE_HOME"
exec "$E4_PROJECT/envs/e4_20261123/bin/python" -u \
  "$E4_PROJECT/releases/$E4_RELEASE/lane.py" \
  --root "$E4_PROJECT/releases/$E4_RELEASE" --project "$E4_PROJECT"
