"""Batched native C4 rollouts with paired initial states and resumable records."""

import argparse
from collections import deque
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from . import ROOT
from .env import coverage
from .env_c4 import PushTC4Env
from .geometry import normalize_xy, denormalize_xy
from .models import make_model
from .train import atomic_json, make_scheduler


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def predict_batch(model, config, histories, device, generators):
    images = torch.tensor(np.stack([[obs["image"] for obs in history] for history in histories]),
                          dtype=torch.float32, device=device)
    positions = np.stack([[obs["agent_pos"] for obs in history] for history in histories])
    positions = torch.tensor(normalize_xy(positions), dtype=torch.float32, device=device)
    sample = randn_tensor((len(histories), config["horizon"], 2), generator=generators,
                          device=torch.device(device), dtype=torch.float32)
    scheduler = make_scheduler(config)
    scheduler.set_timesteps(config["inference_steps"], device=device)
    encoded = model.encode(images, positions)
    for timestep in scheduler.timesteps:
        noise = model.predict_encoded(sample, timestep, encoded)
        sample = scheduler.step(noise, timestep, sample, generator=generators).prev_sample
    start = config["obs_steps"] - 1
    return denormalize_xy(sample[:, start:start + config["action_steps"]].cpu().numpy())


def rollout_batch(model, config, tasks, device, max_steps):
    envs, histories, generators = [], [], []
    values = [[] for _ in tasks]
    active = list(range(len(tasks)))
    try:
        for seed, orientation in tasks:
            env = PushTC4Env(orientation=orientation, recenter=True, legacy=True)
            envs.append(env)
            env.seed(seed)
            obs = env.reset()
            histories.append(deque([obs] * config["obs_steps"], maxlen=config["obs_steps"]))
            generators.append(torch.Generator(device=device).manual_seed(seed + 1000003 * orientation))
        while active:
            actions = predict_batch(model, config, [histories[i] for i in active], device,
                                    [generators[i] for i in active])
            next_active = []
            for i, sequence in zip(active, actions):
                done = False
                for action in sequence:
                    obs, _, done, _ = envs[i].step(action)
                    histories[i].append(obs)
                    values[i].append(coverage(envs[i]))
                    if done or len(values[i]) >= max_steps:
                        break
                if not done and len(values[i]) < max_steps:
                    next_active.append(i)
            active = next_active
        return [{"base_seed": seed, "orientation": k, "policy_seed": seed + 1000003 * k,
                 "max_coverage": max(v), "final_coverage": v[-1],
                 "success": bool(max(v) > 0.95), "steps": len(v)}
                for (seed, k), v in zip(tasks, values)]
    finally:
        for env in envs:
            env.close()


def summarize(rows):
    def metrics(items):
        return {"episodes": len(items),
                "mean_max_coverage": float(np.mean([r["max_coverage"] for r in items])),
                "success_rate": float(np.mean([r["success"] for r in items]))}
    return {**metrics(rows), "per_orientation": {
        str(k): metrics([r for r in rows if r["orientation"] == k])
        for k in sorted({r["orientation"] for r in rows})}}


def evaluate(checkpoint, output, validation_report, device="cuda", batch_size=8,
             max_steps=300, split="validation", base_count=None, resume=False):
    if batch_size < 1 or max_steps < 1:
        raise ValueError("Positive batch_size and max_steps required")
    seeds = json.loads((ROOT / "manifests/evaluation_seeds.json").read_text())
    selected = seeds["test_base_seeds" if split == "test" else "validation"]
    if split == "test" and (base_count is not None or max_steps != 300):
        raise ValueError("The reserved test protocol requires all 25 seeds and 300 steps")
    if base_count is not None:
        if not 1 <= base_count <= len(selected):
            raise ValueError("Invalid base_count")
        selected = selected[:base_count]
    proof = json.loads(Path(validation_report).read_text())
    env_hash = file_hash(ROOT / "src/pusht_study/env_c4.py")
    if not (proof["passed"] and proof["full_validation_episodes"] and
            True in proof.get("legacy_modes", []) and proof["source_sha256"] == env_hash):
        raise ValueError("Require passing full replay validation of the current environment")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, k) for seed in selected for k in seeds["orientations"]]
    signature = {"checkpoint_sha256": file_hash(checkpoint), "environment_sha256": env_hash,
                 "evaluator_sha256": file_hash(__file__),
                 "model_source_sha256": {name: file_hash(ROOT / "src/pusht_study" / name)
                                          for name in ("models.py", "equidiff.py", "geometry.py", "train.py")},
                 "split": split, "base_seeds": selected, "orientations": seeds["orientations"],
                 "max_steps": max_steps, "batch_size": batch_size, "tf32": False,
                 "torch": torch.__version__, "device": device}
    rows = []
    if output.exists():
        previous = json.loads(output.read_text())
        if not resume or previous["signature"] != signature:
            raise FileExistsError("Existing evaluation; resume requires an identical protocol")
        rows = previous["episode_results"]
        if previous["state"] == "complete":
            return previous
    done = {(row["base_seed"], row["orientation"]) for row in rows}
    if len(done) != len(rows) or not done.issubset(set(tasks)):
        raise ValueError("Invalid resumed episode records")
    if len(done) == len(tasks):
        previous["state"] = "complete"
        atomic_json(output, previous)
        return previous
    pending = [task for task in tasks if task not in done]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config, checkpoint_step = saved["config"], saved["step"]
    torch.set_num_threads(config.get("cpu_threads", 4))
    model = make_model(config).to(device)
    model.load_state_dict(saved["ema"])
    del saved
    model.eval()
    begin = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        rows.extend(rollout_batch(model, config, pending[start:start + batch_size], device, max_steps))
        result = {"state": "complete" if len(rows) == len(tasks) else "evaluating",
                  "protocol": "native_c4_canonical_raster_v1", "signature": signature,
                  "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": checkpoint_step,
                  "method": config["method"], "training_seed": config["seed"], "demos": config["demos"],
                  "episode_results": sorted(rows, key=lambda r: (r["base_seed"], r["orientation"])),
                  "summary": summarize(rows), "attempt_seconds": time.perf_counter() - begin,
                  "goal_convention": proof["goal_convention"], "success_definition": "coverage > 0.95"}
        atomic_json(output, result)
        print(json.dumps({"completed": len(rows), "total": len(tasks),
                          "seconds": result["attempt_seconds"]}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--validation-report", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--base-count", type=int)
    p.add_argument("--resume", action="store_true")
    args = vars(p.parse_args())
    Path(args["output"]).parent.mkdir(parents=True, exist_ok=True)
    with open(args["output"] + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = evaluate(**args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
