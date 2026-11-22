"""Preflight a frozen migrated lane, then use the existing archive-gated runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for relative, expected in json.loads((root / "manifests/frozen_files.json").read_text()).items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    migration = json.loads((root / "manifests/migration.json").read_text())
    imported = migration.get("imported_checkpoint")
    if imported:
        checkpoint = root / "runs/equidiff_c4_n25_s0/last.pt"
        # An absent or partial import must never silently start a fresh seed 0.
        assert checkpoint.is_file() and checkpoint.stat().st_size == imported["bytes"]
        checksum = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        assert checksum.hexdigest() == imported["sha256"]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), SDL_VIDEODRIVER="dummy",
               PYGAME_HIDE_SUPPORT_PROMPT="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    with (root / "logs/full_gpu_preflight.log").open("a") as log:
        subprocess.run([sys.executable, str(root / "scripts/check_gpu.py"), "--root", str(root)],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    runner = root.parents[1] / "operations/parallel_20261119/convergence.py"
    os.execve(sys.executable, [sys.executable, str(runner), "--root", str(root)], env)


if __name__ == "__main__":
    main()
