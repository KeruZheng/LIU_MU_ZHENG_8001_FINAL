"""Validate and unpack the official archive; freeze trajectory-level splits."""

import hashlib
import json
from pathlib import Path
import zipfile
from pusht_study import ROOT
from pusht_study.data import dataset_metadata, make_split
from pusht_study.train import atomic_json


def main():
    archive = ROOT / "data/pusht.zip"
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        if not partial.exists():
            raise FileNotFoundError("Download official data to data/pusht.zip.part first")
        with zipfile.ZipFile(partial) as stream:
            if stream.testzip() is not None:
                raise ValueError("Corrupt dataset archive")
        partial.rename(archive)
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe dataset archive path")
        if not (ROOT / "data/pusht/pusht_cchi_v7_replay.zarr").exists():
            stream.extractall(ROOT / "data")
    path = ROOT / "data/pusht/pusht_cchi_v7_replay.zarr"
    metadata = dataset_metadata(path)
    metadata.update(url="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip",
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    archive_bytes=archive.stat().st_size, path=str(path.relative_to(ROOT)))
    atomic_json(ROOT / "manifests/data.json", metadata)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    hashes = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}" for p in files]
    (ROOT / "manifests/data.sha256").write_text("\n".join(hashes) + "\n")
    split = make_split(metadata["episode_count"])
    split_path = ROOT / "manifests/split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("Refusing to overwrite a different existing split")
    atomic_json(split_path, split)
    atomic_json(ROOT / "manifests/evaluation_seeds.json", {
        "validation": list(range(900000, 900010)),
        "test_base_seeds": list(range(100000, 100025)),
        "orientations": [0, 1, 2, 3],
        "status": "reserved; native rotated protocol not yet validated",
    })
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
