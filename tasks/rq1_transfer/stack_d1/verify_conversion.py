"""Verify an official conversion completely before publishing its input path."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


RAW_SHA256 = "edc9815a2738a583dcb3e7a9e239a71eca6057396a839e52b022935c273cfd61"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source, partial, output):
    if output.exists() or not partial.name.endswith(".part"):
        raise ValueError("Use an existing .part conversion and a new final output path")
    source_hash = sha256(source)
    assert source_hash == RAW_SHA256, "Original official dataset checksum changed"
    arrays = 0
    frames = 0
    with h5py.File(source, "r") as src, h5py.File(partial, "r") as dst:
        names = []
        src.visit(names.append)
        other = []
        dst.visit(other.append)
        assert names == other
        for name in ["/"] + names:
            left, right = src[name], dst[name]
            assert set(left.attrs) == set(right.attrs), name
            for key in left.attrs:
                assert np.array_equal(left.attrs[key], right.attrs[key]), (name, key)
            if not isinstance(left, h5py.Dataset):
                continue
            assert left.shape == right.shape and left.dtype == right.dtype, name
            a, b = left[:], right[:]
            if name.endswith("/actions"):
                assert b.ndim == 2 and b.shape[1] == 7, name
                assert np.isfinite(b).all(), name
                assert np.array_equal(a[:, -1], b[:, -1]), name
                assert not np.array_equal(a[:, :6], b[:, :6]), name
                frames += len(b)
            else:
                assert np.array_equal(a, b), name
            arrays += 1
        assert len(src["data"]) == 1000 and frames == 108233
    converted_hash = sha256(partial)
    receipt = {
        "source": source.name, "source_sha256": source_hash,
        "output": output.name, "output_sha256": converted_hash,
        "output_bytes": partial.stat().st_size, "demos": 1000, "frames": frames,
        "verified_arrays": arrays, "unchanged_non_action_arrays": True,
        "unchanged_metadata": True, "finite_absolute_actions": True,
        "unchanged_gripper_actions": True,
        "converter": "upstream/equidiff/equi_diffpo/scripts/robomimic_dataset_conversion.py",
        "workers": 2,
        "action_semantics": "absolute xyz, axis-angle, gripper; original env_args retained as in upstream conversion",
    }
    partial.rename(output)
    output.with_suffix(".conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source, args.partial, args.output)
