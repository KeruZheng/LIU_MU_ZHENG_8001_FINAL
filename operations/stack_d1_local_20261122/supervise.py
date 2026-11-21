"""Singleton local owner: preflight must succeed before formal Stack-D1 starts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    status = {"supervisor_pid": os.getpid(), "run": "stack_d1_n100_s0"}

    def update(**kwargs):
        status.update(kwargs, heartbeat=time.time())
        path = root / "results/supervisor_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(path)

    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled",
               PYTHONPATH=str(root / "upstream"))
    with (root / "results/compute.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for mode in ("preflight", "train"):
                with (root / "logs" / f"{mode}.log").open("x") as log:
                    child = subprocess.Popen([sys.executable, "-u", str(root / "run.py"),
                                              "--root", str(root), "--mode", mode],
                                             cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    update(state=mode, child_pid=child.pid)
                    while child.poll() is None:
                        update()
                        time.sleep(10)
                    if child.returncode:
                        raise RuntimeError(f"{mode} exited {child.returncode}")
                if mode == "preflight":
                    assert json.loads((root / "preflight/result.json").read_text())["state"] == "complete"
            update(state="complete", child_pid=None)
        except BaseException as exc:
            update(state="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
