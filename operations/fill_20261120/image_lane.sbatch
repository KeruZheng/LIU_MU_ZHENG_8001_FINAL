#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
set -euo pipefail
umask 0002
ROBOTIC_ROOT=${1:?Project root required}
ROBOTIC_RELEASE=${2:?Release required}
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
exec "$ROBOTIC_ROOT/envs/rq3/bin/python" "$ROBOTIC_ROOT/releases/$ROBOTIC_RELEASE/scripts/image_lane.py" \
  --root "$ROBOTIC_ROOT/releases/$ROBOTIC_RELEASE"
