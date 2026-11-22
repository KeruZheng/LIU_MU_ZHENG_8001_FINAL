"""Finish a checkpoint upload, verify it remotely, then release its held job."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE_VERIFY = r'''
import hashlib,json,pathlib,subprocess,sys
x=json.load(sys.stdin)
root=pathlib.Path(x['remote_root'])
run=root/'releases'/x['release']/'runs/equidiff_c4_n25_s0'
partial=run/'import_last.pt.part'; final=run/'last.pt'
p=partial if partial.exists() else final
assert p.stat().st_size==x['bytes']
h=hashlib.sha256()
with p.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==x['sha256']
if p==partial:
    assert not final.exists()
    partial.rename(final)
proof=dict(x,state='verified')
(run/'checkpoint_import_verified.json').write_text(json.dumps(proof,indent=2)+'\n')
job=subprocess.check_output(['scontrol','show','job',x['job_id'],'-o'],text=True)
assert 'JobName=robotic_c4_mig_b ' in job
if 'Reason=JobHeldUser ' in job:
    subprocess.run(['scontrol','release',x['job_id']],check=True)
else:
    assert 'JobState=RUNNING ' in job or 'JobState=PENDING ' in job,job
print(json.dumps(proof))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)

    def update(state, **kwargs):
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(dict(state=state, heartbeat=time.time(),
                                               job_id=cfg["job_id"], **kwargs), indent=2) + "\n")
        temporary.replace(output / "status.json")

    try:
        update("checking_local_checkpoint")
        source = Path(cfg["source"])
        checksum = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        assert source.stat().st_size == cfg["bytes"] and checksum.hexdigest() == cfg["sha256"]
        target = cfg["ssh"][-1] + ":" + cfg["remote_root"] + "/releases/" + cfg["release"] + "/runs/equidiff_c4_n25_s0/import_last.pt.part"
        command = ["rsync", "-a", "--partial", "--append-verify", "--info=progress2",
                   "-e", shlex.join(cfg["ssh"][:-1]), str(source), target]
        for attempt in range(1, 4):
            with (output / "transfer.log").open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    update("uploading_checkpoint", attempt=attempt, transfer_pid=process.pid)
                    time.sleep(10)
            if process.returncode == 0:
                break
        else:
            raise RuntimeError("Checkpoint upload failed three times; job remains held")
        update("verifying_remote_checkpoint")
        request = {key: cfg[key] for key in ("remote_root", "release", "job_id", "bytes", "sha256")}
        result = subprocess.run(cfg["ssh"] + [shlex.join(["python3", "-c", REMOTE_VERIFY])],
                                input=json.dumps(request), text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        (output / "verified_receipt.json").write_text(result.stdout)
        update("verified_and_job_released")
        # This finite deployment service stops after the scheduler takes ownership.
        fleet_path = Path(cfg["fleet_config"])
        fleet = json.loads(fleet_path.read_text())
        fleet["services"] = [s for s in fleet["services"] if s["name"] != cfg["service_name"]]
        temporary = fleet_path.with_suffix(".stage.tmp")
        temporary.write_text(json.dumps(fleet, indent=2) + "\n")
        temporary.replace(fleet_path)
    except BaseException as exc:
        update("failed_job_remains_held", errors=[repr(exc)])
        raise


if __name__ == "__main__":
    main()
