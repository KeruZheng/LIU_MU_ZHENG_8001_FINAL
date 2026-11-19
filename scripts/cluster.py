"""Explicit cluster check/upload/preflight actions; no implicit formal submission."""

import argparse
from datetime import datetime, timezone
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "upload", "preflight", "submit"])
    parser.add_argument("--cluster", choices=["sicc", "legacy"], default="sicc")
    parser.add_argument("--execute", action="store_true", help="Otherwise print commands only")
    parser.add_argument("--config", help="Relative training config for submit")
    parser.add_argument("--run", help="Unique run name for submit")
    parser.add_argument("--script", default="slurm/pilot.sbatch", help="Slurm script for submit")
    args = parser.parse_args()
    local_config = ROOT / "configs/clusters.local.json"
    config_path = local_config if local_config.exists() else ROOT / "configs/clusters.json"
    cfg = json.loads(config_path.read_text())[args.cluster]
    if "@" not in cfg["host"]:
        raise ValueError("Cluster host must include its configured username: user@host")
    user = cfg["host"].split("@", 1)[0]
    if args.execute and cfg.get("status") == "example_requires_configuration":
        raise SystemExit("Copy configs/clusters.json to configs/clusters.local.json and configure your account")
    remote = cfg["root"]
    if Path(remote).name != "robotic" or not remote.startswith("/"):
        raise ValueError("Remote root must be an absolute robotic directory")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *cfg["ssh_options"]]
    r = shlex.quote(remote)
    commands = []
    if args.action == "check":
        commands.append([*ssh, cfg["host"],
            f"hostname; id; pwd; ls -ld {shlex.quote(str(Path(remote).parent))}; "
            f"squeue -u {shlex.quote(user)} -o '%.18i %.30j %.12P %.2t %.10M %R'; "
            "sinfo -o '%P %a %l %D %G'; quota -s"])
    elif args.action == "upload":
        # Never update source under an active robotic job.
        guard = (f"robotic_jobs=$(squeue -u {shlex.quote(user)} -h -o '%j'); "
                 "if printf '%s\\n' \"$robotic_jobs\" | grep -q '^robotic_'; then "
                 "echo 'Active robotic job: refuse source update' >&2; exit 3; fi")
        setup = f"set -eu; {guard}; mkdir -p {r}/logs/slurm {r}/data {r}/envs {r}/results {r}/runs"
        if cfg["group"]:
            setup += f"; chgrp {shlex.quote(cfg['group'])} {r}; chmod g+s {r}"
        commands.append([*ssh, cfg["host"], setup])
        commands.append(["rsync", "-a", "--no-owner", "--no-group", "--partial",
            "--exclude=.git/", "--exclude=__pycache__/", "--exclude=envs/local/",
            "--exclude=envs/pip-cache/", "--exclude=envs/huggingface/",
            "--exclude=runs/", "--exclude=results/", "--exclude=*.part",
            "-e", shlex.join(ssh), str(ROOT) + "/", cfg["host"] + ":" + remote + "/"])
        if cfg["group"]:
            commands.append([*ssh, cfg["host"],
                f"set -eu; chgrp -R {shlex.quote(cfg['group'])} {r}; "
                f"find {r} -type d -exec chmod g+s {{}} +"])
        commands.append([*ssh, cfg["host"],
                         f"cd {r} && sha256sum --check manifests/code.sha256 manifests/data.sha256"])
    elif args.action == "submit":
        if not args.config or not args.run:
            raise SystemExit("submit requires --config and --run")
        config, run, script = args.config, args.run, args.script
        if config.startswith("/") or ".." in Path(config).parts:
            raise ValueError("Config must be a relative path without '..'")
        if script.startswith("/") or ".." in Path(script).parts:
            raise ValueError("Script must be a relative path without '..'")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
            raise ValueError("Run name may contain only letters, digits, '_' and '-'")
        options = ["sbatch", "--parsable", "--job-name=robotic_" + run,
                   "--partition=" + cfg["partition"],
                   "--cpus-per-task=" + str(cfg["cpus"]), "--mem=" + cfg["memory"],
                   "--output=" + remote + "/logs/slurm/" + run + "_%j.out"]
        for key in ("account", "qos"):
            if cfg[key]:
                options.append("--" + key + "=" + cfg[key])
        options.extend([remote + "/" + script, remote, config, run])
        command = f"set -eu; cd {r}; test -x envs/robotic/bin/python; "
        command += f"test -f {shlex.quote(config)}; "
        command += f"robotic_jobs=$(squeue -u {shlex.quote(user)} -h -o '%j'); "
        command += (f"if printf '%s\\n' \"$robotic_jobs\" | grep -Fqx 'robotic_{run}'; then "
                    "echo 'Job already queued' >&2; exit 3; fi; ")
        command += shlex.join(options)
        commands.append([*ssh, cfg["host"], command])
    else:
        options = ["sbatch", "--parsable", "--job-name=robotic_preflight", "--partition=" + cfg["partition"],
                   "--cpus-per-task=" + str(cfg["cpus"]), "--mem=" + cfg["memory"],
                   "--output=" + remote + "/logs/slurm/preflight_%j.out"]
        for key in ("account", "qos"):
            if cfg[key]:
                options.append("--" + key + "=" + cfg[key])
        options.extend([remote + "/slurm/preflight.sbatch", remote])
        command = f"set -eu; cd {r}; test -x envs/robotic/bin/python; "
        command += (f"robotic_jobs=$(squeue -u {shlex.quote(user)} -h -o '%j'); "
                    "if printf '%s\\n' \"$robotic_jobs\" | grep -q '^robotic_preflight$'; then "
                    "echo 'Preflight already queued' >&2; exit 3; fi; ")
        command += shlex.join(options)
        commands.append([*ssh, cfg["host"], command])
    log = ROOT / "manifests/cluster_actions.jsonl"
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            intent = {"time": datetime.now(timezone.utc).isoformat(), "cluster": args.cluster,
                      "action": args.action, "command": command, "state": "intent"}
            with log.open("a") as stream:
                stream.write(json.dumps(intent) + "\n")
            result = subprocess.run(command, text=True, capture_output=True)
            print(result.stdout, end="")
            print(result.stderr, end="")
            with log.open("a") as stream:
                stream.write(json.dumps({**intent, "state": "returned", "returncode": result.returncode,
                                         "stdout": result.stdout, "stderr": result.stderr}) + "\n")
            if result.returncode:
                raise SystemExit("Action failed; check remote state before retrying submission")


if __name__ == "__main__":
    main()
