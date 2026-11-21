"""Resume the preserved C4 lane only after Stack-D1 completes and verifies."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-root", required=True, type=Path)
    parser.add_argument("--pause-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def update(state, **kwargs):
        destination = args.output / "status.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(state=state, heartbeat=time.time(), **kwargs), indent=2) + "\n")
        temporary.replace(destination)

    run = args.stack_root / "runs/stack_d1_n100_s0"
    with (args.output / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "resumed.json").exists():
            return
        while True:
            supervisor = json.loads((args.stack_root / "results/supervisor_status.json").read_text())
            if supervisor["state"] == "failed":
                update("blocked_stack_failure", supervisor=supervisor)
                raise RuntimeError("Stack-D1 failed; C4 remains paused for investigation")
            if supervisor["state"] == "complete":
                break
            progress = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
            update("waiting_for_stack_d1", stack_progress=progress)
            time.sleep(30)
        update("verifying_stack_artifacts")
        completed = json.loads((run / "completed.json").read_text())
        assert completed["epoch"] == 500 and completed["state"] == "complete"
        assert (run / "final_evaluation.json").is_file()
        inventory = json.loads((run / "local_inventory.json").read_text())
        assert inventory["files"]
        for record in inventory["files"]:
            path = run / "checkpoints" / record["name"]
            assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
        pause = json.loads(args.pause_receipt.read_text())
        assert digest(Path(pause["checkpoint"])) == pause["sha256"]
        command = pause["resume_command"]
        lane = Path(command[-1])
        for relative, expected in json.loads((lane / "manifests/frozen_files.json").read_text()).items():
            assert digest(lane / relative) == expected
        with (args.output / "c4_supervisor.log").open("a") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        result = {"pid": process.pid, "time": time.time(), "from_step": pause["checkpoint_step"],
                  "stack_artifacts_verified": True}
        (args.output / "resumed.json").write_text(json.dumps(result, indent=2) + "\n")
        update("c4_resumed", **result)


if __name__ == "__main__":
    main()
