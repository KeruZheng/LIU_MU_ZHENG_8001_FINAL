"""Fixed-update pilot trainer using the upstream DP temporal U-Net."""

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from . import ROOT
from .data import PushTSequences
from .geometry import rotate_image, rotate_xy
from .models import make_model


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def make_scheduler(config):
    return DDPMScheduler(num_train_timesteps=config["diffusion_steps"],
                         beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                         prediction_type="epsilon", clip_sample=True)


def train(config, output, device="cuda", resume=False):
    if config.get("formal", False):
        raise ValueError("Formal runs are locked pending GPU calibration and protocol review")
    if config.get("rotated", False):
        raise ValueError("Rotated training is locked pending aligned data and native symmetry validation")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback for GPU jobs")
    if config["steps"] <= 0 or config["batch_size"] <= 0:
        raise ValueError("steps and batch_size must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    if (checkpoint.exists() or (output / "metrics.jsonl").exists()) and not resume:
        raise FileExistsError("Existing run; use --resume or a new output directory")
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    split_path = ROOT / config["split"]
    split = json.loads(split_path.read_text())
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    dataset = PushTSequences(ROOT / config["data"], split, config["demos"],
                             config["horizon"], config["obs_steps"], config["action_steps"],
                             config["rotated"])
    device = torch.device(device)
    model = make_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = make_scheduler(config)
    data_rng = torch.Generator().manual_seed(seed + 1000)
    start_step = 0
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {k: v for k, v in config.items() if k != "steps"}
        stored = {k: v for k, v in state["config"].items() if k != "steps"}
        if expected != stored or state["split_sha256"] != split_sha:
            raise ValueError("Resume configuration or split differs from checkpoint")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.to(device)
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        data_rng.set_state(state["data_rng"])
        start_step = state["step"]
    atomic_json(output / "config.json", config)
    atomic_json(output / "runtime.json", {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
        "split_sha256": split_sha, "source": "pilot DP-style implementation; not full paper reproduction",
        "parameters": sum(p.numel() for p in model.parameters()),
        "upstream": json.loads((ROOT / "manifests/upstream.json").read_text()),
        "dataset": json.loads((ROOT / "manifests/data.json").read_text()),
        "code_manifest_sha256": hashlib.sha256((ROOT / "manifests/code.sha256").read_bytes()).hexdigest()
            if (ROOT / "manifests/code.sha256").exists() else None,
    })
    begin = time.perf_counter()
    for step in range(start_step + 1, config["steps"] + 1):
        indices = torch.randint(len(dataset), (config["batch_size"],), generator=data_rng)
        examples = [dataset[int(i)] for i in indices]
        batch = {k: torch.stack([x[k] for x in examples]).to(device)
                 for k in ("image", "agent_pos", "action")}
        if config["method"] == "dp_c4_aug":
            for i in range(config["batch_size"]):
                k = int(torch.randint(4, (), generator=data_rng))
                batch["image"][i] = rotate_image(batch["image"][i], k)
                batch["agent_pos"][i] = rotate_xy(batch["agent_pos"][i], k)
                batch["action"][i] = rotate_xy(batch["action"][i], k)
        noise = torch.randn_like(batch["action"])
        timestep = torch.randint(config["diffusion_steps"], (config["batch_size"],), device=device)
        noisy = scheduler.add_noise(batch["action"], noise, timestep)
        prediction = model(noisy, timestep, batch["image"], batch["agent_pos"])
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        decay = min(0.9999, 1 - (1 + step) ** -0.75)
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1 - decay)
            for target, source in zip(ema.buffers(), model.buffers()):
                target.copy_(source)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"step": step, "loss": loss.item(), "elapsed_seconds": time.perf_counter() - begin,
               "attempt_start_step": start_step,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        atomic_json(output / "status.json", {"state": "training", **row})
        if step % config["checkpoint_every"] == 0 or step == config["steps"]:
            state = {"step": step, "config": config, "split_sha256": split_sha,
                     "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                     "data_rng": data_rng.get_state()}
            temp = output / "last.pt.tmp"
            torch.save(state, temp)
            temp.replace(checkpoint)
        print(json.dumps(row), flush=True)
    atomic_json(output / "status.json", {"state": "complete", "step": max(start_step, config["steps"]),
                                         "checkpoint": str(checkpoint)})
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    # Refuse concurrent writers, including accidental duplicate cluster submissions.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            train(config, output, args.device, args.resume)
        except Exception as exc:
            atomic_json(output / "status.json", {"state": "failed", "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
