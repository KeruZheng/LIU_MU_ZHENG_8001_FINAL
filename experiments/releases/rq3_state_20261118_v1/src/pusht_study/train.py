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
from .data import PushTSequences, PushTStateSequences
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
    if config.get("formal", False) and not config.get("allow_formal", False):
        raise ValueError("Formal runs are locked; set allow_formal after protocol review")
    # `rotated` means each demo gets a fixed base orientation (balanced C4 training data).
    # It is distinct from native rotated evaluation, which remains unimplemented.
    if config.get("rotated", False) and not config.get("allow_rotated", False):
        raise ValueError("Rotated training is locked; set allow_rotated after validating base-orientation data")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback for GPU jobs")
    if device.startswith("cuda") and os.environ.get("ROBOTIC_GPU_MEMORY_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["ROBOTIC_GPU_MEMORY_FRACTION"]))
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
    torch.backends.cuda.matmul.allow_tf32 = config.get("allow_tf32", False)
    torch.backends.cudnn.allow_tf32 = config.get("allow_tf32", False)
    split_path = ROOT / config["split"]
    split = json.loads(split_path.read_text())
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    state_input = config.get("observation") == "state"
    if state_input:
        if config["rotated"] != (config.get("goal_mode") == "rotated"):
            raise ValueError("State data rotation must agree with the goal mode")
        dataset = PushTStateSequences(ROOT / config["data"], split, config["demos"],
                                     config["horizon"], config["obs_steps"], config["action_steps"], config["rotated"])
    else:
        dataset = PushTSequences(ROOT / config["data"], split, config["demos"],
                                 config["horizon"], config["obs_steps"], config["action_steps"],
                                 config["rotated"], cache=config.get("cache_data", False))
    device = torch.device(device)
    model = make_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if getattr(model, "uses_escnn", False):
        # escnn caches expanded filters in eval mode; EMA stores trainable coefficients.
        ema.train()
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
        "split_sha256": split_sha,
        "source": ("RQ3 state-input controlled symmetry study" if state_input else
                   "Push-T adaptation of pinned EquiDiff architecture" if config["method"] == "equidiff_c4"
                   else "pilot DP-style implementation; not full paper reproduction"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "allow_tf32": config.get("allow_tf32", False),
        "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*.py"))},
        "upstream": json.loads((ROOT / "manifests/upstream.json").read_text()),
        "dataset": json.loads((ROOT / "manifests/data.json").read_text()),
        "code_manifest_sha256": hashlib.sha256((ROOT / "manifests/code.sha256").read_bytes()).hexdigest()
            if (ROOT / "manifests/code.sha256").exists() else None,
    })
    begin = time.perf_counter()
    for step in range(start_step + 1, config["steps"] + 1):
        step_begin = time.perf_counter()
        indices = torch.randint(len(dataset), (config["batch_size"],), generator=data_rng)
        batch = {k: value.to(device) for k, value in dataset.get_batch(indices).items()}
        data_seconds = time.perf_counter() - step_begin
        if config["method"] == "dp_c4_aug":
            orientations = torch.randint(4, (config["batch_size"],), generator=data_rng).to(device)
            for k in range(1, 4):
                mask = orientations == k
                batch["image"][mask] = rotate_image(batch["image"][mask], k)
                batch["agent_pos"][mask] = rotate_xy(batch["agent_pos"][mask], k)
                batch["action"][mask] = rotate_xy(batch["action"][mask], k)
        noise = torch.randn_like(batch["action"])
        timestep = torch.randint(config["diffusion_steps"], (config["batch_size"],), device=device)
        noisy = scheduler.add_noise(batch["action"], noise, timestep)
        prediction = (model(noisy, timestep, batch["state"], batch["goal"]) if state_input
                      else model(noisy, timestep, batch["image"], batch["agent_pos"]))
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
            source_buffers = dict(model.named_buffers())
            for name, target in ema.named_buffers():
                target.copy_(source_buffers[name])
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"step": step, "loss": loss.item(), "elapsed_seconds": time.perf_counter() - begin,
               "data_seconds": data_seconds, "update_seconds": time.perf_counter() - step_begin,
               "attempt_start_step": start_step,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        if hasattr(model, "symmetry_gate"):
            row["symmetry_gate"] = model.symmetry_gate.item()
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
        if step in config.get("retain_ema_steps", []):
            retained = {"step": step, "config": config, "split_sha256": split_sha,
                        "ema": ema.state_dict(), "checkpoint_kind": "evaluation_only"}
            temp = output / f"ema_step_{step}.pt.tmp"
            torch.save(retained, temp)
            temp.replace(output / f"ema_step_{step}.pt")
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
