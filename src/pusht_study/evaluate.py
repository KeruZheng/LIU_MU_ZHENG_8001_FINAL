import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from .env import coverage
from .geometry import normalize_xy, denormalize_xy
from .models import make_model
from .train import make_scheduler, atomic_json


@torch.no_grad()
def predict(model, config, observations, device, generator):
    images = torch.tensor(np.stack([o["image"] for o in observations]), device=device).unsqueeze(0)
    agent = torch.tensor(normalize_xy(np.stack([o["agent_pos"] for o in observations])),
                         dtype=torch.float32, device=device).unsqueeze(0)
    sample = torch.randn((1, config["horizon"], 2), device=device, generator=generator)
    scheduler = make_scheduler(config)
    scheduler.set_timesteps(config["inference_steps"], device=device)
    encoded = model.encode(images, agent)
    for timestep in scheduler.timesteps:
        noise = model.predict_encoded(sample, timestep, encoded)
        sample = scheduler.step(noise, timestep, sample, generator=generator).prev_sample
    start = config["obs_steps"] - 1
    return denormalize_xy(sample[0, start:start + config["action_steps"]].cpu().numpy())


def evaluate(checkpoint, output, device="cuda", episodes=2, max_steps=30, seed_start=900000):
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    if config["rotated"]:
        raise ValueError("Native rotated evaluation is not yet validated")
    torch.set_num_threads(config.get("cpu_threads", 4))
    model = make_model(config).to(device).eval()
    model.load_state_dict(state["ema"])
    rows = []
    for seed in range(seed_start, seed_start + episodes):
        env = PushTImageEnv(legacy=True)
        env.seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        try:
            obs = env.reset()
            history = deque([obs] * config["obs_steps"], maxlen=config["obs_steps"])
            values = []
            done = False
            while len(values) < max_steps and not done:
                actions = predict(model, config, history, device, generator)
                for action in actions:
                    obs, _, done, _ = env.step(action)
                    history.append(obs)
                    values.append(coverage(env))
                    if done or len(values) >= max_steps:
                        break
            rows.append({"seed": seed, "max_coverage": max(values), "final_coverage": values[-1],
                         "success": bool(max(values) > 0.95), "steps": len(values)})
        finally:
            env.close()
    result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": state["step"],
              "protocol": "original_pusht_pilot", "episodes": rows,
              "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
              "success_rate": float(np.mean([r["success"] for r in rows]))}
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=900000)
    a = p.parse_args()
    print(json.dumps(evaluate(a.checkpoint, a.output, a.device, a.episodes, a.max_steps, a.seed_start), indent=2))
