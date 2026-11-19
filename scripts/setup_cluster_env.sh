#!/bin/bash
set -euo pipefail
ROBOTIC_ROOT=${1:?Absolute robotic project directory required}
ROBOTIC_BASE_PYTHON=${2:-python3.10}
[[ "$ROBOTIC_ROOT" == /* && "${ROBOTIC_ROOT##*/}" == robotic ]]
cd "$ROBOTIC_ROOT"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ROBOTIC_ROOT/envs/pip-cache"
export UV_CACHE_DIR="$ROBOTIC_ROOT/envs/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROBOTIC_ROOT/envs/python"
export UV_PYTHON_BIN_DIR="$ROBOTIC_ROOT/envs/bin"
export TMPDIR="$ROBOTIC_ROOT/envs/tmp"
mkdir -p "$TMPDIR" manifests
exec 9>envs/setup.lock
flock -n 9

check_venv() {
  "$1/bin/python" - "$1" <<'PY'
import pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve()
assert expected.is_relative_to(pathlib.Path.cwd().resolve()), "Venv must stay inside robotic"
assert pathlib.Path(sys.prefix).resolve() == expected, (sys.prefix, expected)
assert sys.prefix != sys.base_prefix, "Refuse installation outside a venv"
cfg = (expected / "pyvenv.cfg").read_text().lower()
assert "include-system-site-packages = false" in cfg, "Require an isolated venv"
PY
}

if ! command -v "$ROBOTIC_BASE_PYTHON" >/dev/null 2>&1; then
  # Bootstrap a private interpreter; never install into the login node's Python.
  [[ "$ROBOTIC_BASE_PYTHON" == python3.10 ]]
  if [[ ! -x envs/bootstrap/bin/python ]]; then
    python3 -m venv envs/bootstrap
  fi
  check_venv "$ROBOTIC_ROOT/envs/bootstrap"
  envs/bootstrap/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" \
    install --index-url https://pypi.org/simple uv==0.9.18
  envs/bootstrap/bin/uv python install 3.10
  ROBOTIC_BASE_PYTHON=$(envs/bootstrap/bin/uv python find --managed-python 3.10)
fi

if [[ ! -x envs/robotic/bin/python ]]; then
  "$ROBOTIC_BASE_PYTHON" -m venv envs/robotic
fi
check_venv "$ROBOTIC_ROOT/envs/robotic"
envs/robotic/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
envs/robotic/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" \
  install --index-url https://pypi.org/simple pip==25.3 setuptools==75.8.2 wheel==0.45.1
envs/robotic/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" \
  install --only-binary=:all: torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
envs/robotic/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" \
  install --index-url https://pypi.org/simple -r configs/requirements.txt
envs/robotic/bin/python -m pip check
envs/robotic/bin/python -m pip freeze > manifests/cluster_environment.txt
envs/robotic/bin/python - <<'PY'
from datetime import datetime, timezone
import json, pathlib, platform, sys
pathlib.Path("manifests/cluster_environment.json").write_text(json.dumps({
    "time": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
    "python": sys.version, "executable": sys.executable,
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "state": "installed_pending_slurm_preflight",
}, indent=2) + "\n")
PY
# Run CUDA and model checks only through Slurm after installation.
