"""Check the actual Slurm GPU, full C4 backward, and imported resume state."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert os.environ.get("SLURM_JOB_ID"), "GPU checks must run inside Slurm"
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import torch
    from pusht_study.data import PushTSequences
    from pusht_study.models import make_model
    from pusht_study.train import atomic_json, make_scheduler

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lane = json.loads((root / "manifests/lane.json").read_text())
    run = lane["runs"][0]
    cfg = json.loads((root / "configs" / f"{run}_20000.json").read_text())
    split_path = root / cfg["split"]
    split = json.loads(split_path.read_text())
    model = make_model(cfg).to("cuda")
    ema = copy.deepcopy(model).train().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  betas=(0.95, 0.999), weight_decay=1e-6)
    data_rng = torch.Generator().manual_seed(cfg["seed"] + 1000)
    checkpoint = root / "runs" / run / "last.pt"
    restored_step = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert {k: v for k, v in state["config"].items() if k != "steps"} == {
            k: v for k, v in cfg.items() if k != "steps"}
        assert state["split_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.state.values():
            for key, value in group.items():
                if torch.is_tensor(value):
                    group[key] = value.to("cuda")
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        data_rng.set_state(state["data_rng"])
        restored_step = state["step"]
        assert restored_step == 12000
        del state
    dataset = PushTSequences(root / cfg["data"], split, cfg["demos"], cfg["horizon"],
                             cfg["obs_steps"], cfg["action_steps"], cfg["rotated"], cache=True)
    indices = torch.randint(len(dataset), (cfg["batch_size"],), generator=data_rng)
    batch = {key: value.to("cuda") for key, value in dataset.get_batch(indices).items()}
    scheduler = make_scheduler(cfg)
    noise = torch.randn_like(batch["action"])
    timestep = torch.randint(cfg["diffusion_steps"], (cfg["batch_size"],), device="cuda")
    noisy = scheduler.add_noise(batch["action"], noise, timestep)
    begin = time.monotonic()
    loss = torch.nn.functional.mse_loss(
        model(noisy, timestep, batch["image"], batch["agent_pos"]), noise)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    result = {"state": "complete", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "batch_size": cfg["batch_size"], "restored_step": restored_step,
              "restored_optimizer_and_rng": restored_step is not None,
              "loss": loss.item(), "update_seconds": time.monotonic() - begin,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "job_id": os.environ["SLURM_JOB_ID"],
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope": "One disposable update; production starts separately from the unchanged checkpoint"}
    atomic_json(root / "results/full_gpu_preflight.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
