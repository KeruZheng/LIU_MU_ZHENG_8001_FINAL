"""Paired two-way bootstrap over training seeds and base initial states."""

import numpy as np


def paired_bootstrap(left, right, repetitions=10000, seed=20261118):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 4:
        raise ValueError("Expected matching [training_seed, base_seed, four_orientations] arrays")
    if left.shape[0] < 3 or left.shape[1] < 2 or repetitions < 100:
        raise ValueError("Require three training seeds, paired base seeds, and at least 100 resamples")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite result")
    delta = (left - right).mean(axis=2)
    rng = np.random.default_rng(seed)
    train = rng.integers(delta.shape[0], size=(repetitions, delta.shape[0]))
    bases = rng.integers(delta.shape[1], size=(repetitions, delta.shape[1]))
    estimates = delta[train[:, :, None], bases[:, None, :]].mean(axis=(1, 2))
    return {"mean_difference": float(delta.mean()),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "training_seed_differences": delta.mean(axis=1).tolist(),
            "repetitions": repetitions, "bootstrap_seed": seed,
            "resampling": "Paired training seeds and paired base states; four orientations stay together"}


def evaluation_array(results, metric="max_coverage"):
    if metric not in ("max_coverage", "success"):
        raise ValueError(metric)
    seeds = sorted(r["training_seed"] for r in results)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seeds; average retained checkpoints explicitly first")
    ordered = sorted(results, key=lambda r: r["training_seed"])
    keys = sorted((r["base_seed"], r["orientation"]) for r in ordered[0]["episode_results"])
    bases = sorted({s for s, k in keys})
    if keys != [(s, k) for s in bases for k in range(4)]:
        raise ValueError("Incomplete orientation grid")
    arrays = []
    for result in ordered:
        if result["state"] != "complete":
            raise ValueError("Incomplete evaluation")
        rows = sorted(result["episode_results"], key=lambda r: (r["base_seed"], r["orientation"]))
        if [(r["base_seed"], r["orientation"]) for r in rows] != keys:
            raise ValueError("Unpaired initial states")
        arrays.append(np.array([r[metric] for r in rows], dtype=float).reshape(len(bases), 4))
    return seeds, bases, np.stack(arrays)
