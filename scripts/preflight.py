import argparse
import json
import platform
import sys
import time
import torch
from pusht_study import ROOT
from pusht_study.env import PushTRotEnv
from pusht_study.models import Denoiser, C4FrameAverage
from pusht_study.geometry import rotate_xy, rotate_image
from pusht_study.train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--full-model", action="store_true")
    p.add_argument("--output", default=str(ROOT / "results/preflight.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    info = {"python": sys.version, "torch": torch.__version__, "host": platform.node(),
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "full_model": args.full_model, "formal_ready": False}
    if args.require_cuda and not info["cuda_available"]:
        info["status"] = "blocked_cuda_unavailable"
        atomic_json(args.output, info)
        raise SystemExit("CUDA unavailable")
    device = "cuda" if args.require_cuda else "cpu"
    info["device"] = device
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name()
    env = PushTRotEnv()
    try:
        env.seed(42)
        obs = env.reset()
        env.step(obs["agent_pos"])
        info["observation_shape"] = list(obs["image"].shape)
    finally:
        env.close()
    torch.manual_seed(0)
    model = C4FrameAverage(Denoiser(tiny=not args.full_model)).to(device)
    image = torch.randn(1, 2, 3, 96, 96, device=device)
    pos, action = torch.randn(1, 2, 2, device=device), torch.randn(1, 16, 2, device=device)
    timestep = torch.tensor([3], device=device)
    begin = time.perf_counter()
    prediction = model(action, timestep, image, pos)
    prediction.square().mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    info["forward_backward_seconds"] = time.perf_counter() - begin
    with torch.no_grad():
        actual = model(rotate_xy(action, 1), timestep, rotate_image(image, 1), rotate_xy(pos, 1))
        info["equivariance_max_error"] = (actual - rotate_xy(prediction, 1)).abs().max().item()
    if info["equivariance_max_error"] > 1e-5:
        raise AssertionError(info)
    info["status"] = "passed"
    atomic_json(args.output, info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
