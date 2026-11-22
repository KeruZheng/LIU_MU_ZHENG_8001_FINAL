#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
set -euo pipefail
ROBOTIC_ROOT=${1:?Project root required}
ROBOTIC_FIRST=${2:?First release required}
ROBOTIC_SECOND=${3:?Second release required}
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
# Pin independent steps; otherwise this Slurm version may serialize on node one.
mapfile -t ROBOTIC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
srun --exclusive --exact --nodes=1 --ntasks=1 --nodelist="${ROBOTIC_NODES[0]}" --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  "$ROBOTIC_ROOT/envs/c4_legacy_20261122/bin/python" \
  "$ROBOTIC_ROOT/releases/$ROBOTIC_FIRST/scripts/run_lane.py" \
  --root "$ROBOTIC_ROOT/releases/$ROBOTIC_FIRST" &
ROBOTIC_FIRST_PID=$!
srun --exclusive --exact --nodes=1 --ntasks=1 --nodelist="${ROBOTIC_NODES[-1]}" --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  "$ROBOTIC_ROOT/envs/c4_legacy_20261122/bin/python" \
  "$ROBOTIC_ROOT/releases/$ROBOTIC_SECOND/scripts/run_lane.py" \
  --root "$ROBOTIC_ROOT/releases/$ROBOTIC_SECOND" &
ROBOTIC_SECOND_PID=$!
ROBOTIC_EXIT=0
wait "$ROBOTIC_FIRST_PID" || ROBOTIC_EXIT=1
wait "$ROBOTIC_SECOND_PID" || ROBOTIC_EXIT=1
exit "$ROBOTIC_EXIT"
