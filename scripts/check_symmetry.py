"""Measure native physics and raster errors without treating them as identical."""

import json
import numpy as np
from pusht_study import ROOT
from pusht_study.env import PushTRotEnv, coverage
from pusht_study.geometry import rotate_image, rotate_xy, WORLD_CENTER
from pusht_study.train import atomic_json


def main():
    rows = []
    for seed in range(5):
        envs = [PushTRotEnv(orientation=k, legacy=False) for k in range(4)]
        try:
            observations = []
            for env in envs:
                env.seed(seed)
                observations.append(env.reset())
            rng = np.random.default_rng(seed)
            for step in range(21):
                if step:
                    action = rng.uniform(100, 400, 2)
                    observations = [e.step(rotate_xy(action, k, WORLD_CENTER))[0]
                                    for k, e in enumerate(envs)]
                for k in range(1, 4):
                    pixel = np.abs(rotate_image(observations[0]["image"], k) - observations[k]["image"])
                    position = np.abs(rotate_xy(np.array(envs[0].block.position), k, WORLD_CENTER)
                                      - np.array(envs[k].block.position)).max()
                    rows.append({"seed": seed, "step": step, "orientation": k,
                                 "block_position_max_error": float(position),
                                 "coverage_error": abs(coverage(envs[0]) - coverage(envs[k])),
                                 "pixel_mae": float(pixel.mean()), "pixel_max_error": float(pixel.max())})
        finally:
            for env in envs:
                env.close()
    maxima = {key: max(r[key] for r in rows) for key in
              ("block_position_max_error", "coverage_error", "pixel_mae", "pixel_max_error")}
    report = {"protocol": "5 seeds, 20 control steps, native render; diagnostic only",
              "maxima": maxima, "exact_pixel_equivariance": maxima["pixel_max_error"] == 0,
              "formal_ready": False, "reason": "Need demo replay, rendering alignment and prespecified tolerances",
              "measurements": rows}
    atomic_json(ROOT / "results/symmetry_native.json", report)
    print(json.dumps(maxima, indent=2))


if __name__ == "__main__":
    main()
