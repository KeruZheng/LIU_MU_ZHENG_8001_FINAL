"""Validate native rotated dynamics, canonical rendering, and held-out demo replay."""

import argparse
import hashlib
import json
import time
from itertools import product
import numpy as np
import zarr
from pusht_study import ROOT
from pusht_study.env import coverage
from pusht_study.env_c4 import PushTC4Env
from pusht_study.geometry import WORLD_CENTER, rotate_xy, rotate_image
from pusht_study.train import atomic_json


def validate(output, max_steps=None):
    begin = time.perf_counter()
    split = json.loads((ROOT / "manifests/split.json").read_text())
    data = zarr.open_group(str(ROOT / "data/pusht/pusht_cchi_v7_replay.zarr"), mode="r")
    ends = np.asarray(data["meta/episode_ends"])
    # Fixed before inspecting replay results. Failures remain failures in the report.
    tolerances = dict(position=1e-3, velocity=1e-3, angle=1e-5, coverage=1e-5,
                      pixel_mae=1e-5, pixel_max=1 / 255 + 1e-7)
    maxima = {key: 0.0 for key in tolerances}
    rows = []
    for episode, legacy in product(split["validation"], (False, True)):
        start = int(ends[episode - 1]) if episode else 0
        end = int(ends[episode])
        state = np.asarray(data["data/state"][start])[:5]
        actions = np.asarray(data["data/action"][start:end])
        if max_steps is not None:
            actions = actions[:max_steps]
        envs = [PushTC4Env(orientation=k, legacy=legacy) for k in range(4)]
        episode_max = dict.fromkeys(tolerances, 0.0)
        try:
            obs = []
            for env in envs:
                env.seed(900000 + episode)
                env.canonical_state = state
                obs.append(env.reset())
            for step in range(len(actions) + 1):
                if step:
                    obs = [env.step(rotate_xy(actions[step - 1], k, WORLD_CENTER))[0]
                           for k, env in enumerate(envs)]
                for k in range(1, 4):
                    expected = rotate_xy(np.array(envs[0].block.position), k, WORLD_CENTER)
                    pixel = np.abs(rotate_image(obs[0]["image"], k) - obs[k]["image"])
                    delta_angle = envs[k].block.angle - envs[0].block.angle + k * np.pi / 2
                    errors = {
                        "position": float(np.max(np.abs(expected - envs[k].block.position))),
                        "velocity": float(np.max(np.abs(rotate_xy(np.array(envs[0].block.velocity), k)
                                                          - envs[k].block.velocity))),
                        "angle": float(abs(np.arctan2(np.sin(delta_angle), np.cos(delta_angle)))),
                        "coverage": abs(coverage(envs[0]) - coverage(envs[k])),
                        "pixel_mae": float(pixel.mean()), "pixel_max": float(pixel.max())}
                    for name, value in errors.items():
                        episode_max[name] = max(episode_max[name], value)
            rows.append({"episode": episode, "legacy": legacy, "steps": len(actions), "maxima": episode_max,
                         "final_coverages": [coverage(env) for env in envs]})
            for key in maxima:
                maxima[key] = max(maxima[key], episode_max[key])
        finally:
            for env in envs:
                env.close()
    report = {"protocol": "native_c4_canonical_raster_v1", "tolerances": tolerances,
              "maxima": maxima, "passed": all(maxima[k] <= v for k, v in tolerances.items()),
              "full_validation_episodes": max_steps is None, "legacy_modes": [False, True], "episodes": rows,
              "seconds": time.perf_counter() - begin,
              "source_sha256": hashlib.sha256((ROOT / "src/pusht_study/env_c4.py").read_bytes()).hexdigest(),
              "scope": "Held-out demo action replay across native orientations; not reproduction of recorded trajectories",
              "goal_convention": "Rotate original (256,256) position and angle about (255.5,255.5)",
              "renderer": "Inverse-transform physical state, use original rasterizer, rotate pixels"}
    atomic_json(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    report = validate(args.output, args.max_steps)
    raise SystemExit(0 if report["passed"] else 1)
