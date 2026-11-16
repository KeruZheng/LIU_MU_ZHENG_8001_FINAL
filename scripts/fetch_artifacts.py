"""Restore pinned official sources and optionally the data after a Git clone."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_hash):
    if path.exists():
        if sha256(path) != expected_hash:
            raise ValueError(f"Existing archive has unexpected SHA256: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "15",
                    "--max-time", "1800", url, "-o", str(partial)], check=True)
    if sha256(partial) != expected_hash:
        raise ValueError(f"Downloaded archive has unexpected SHA256: {partial}")
    partial.replace(path)


def restore_sources():
    for item in json.loads((ROOT / "manifests/upstream.json").read_text()):
        name, commit = item["name"], item["commit"]
        repository = item["repository"].removeprefix("https://github.com/")
        destination = ROOT / "upstream" / name
        if destination.exists():
            print(f"Keep existing source: {destination}; verify using manifests/code.sha256")
            continue
        archive = ROOT / "upstream" / f"{name}.tar.gz"
        download(f"https://codeload.github.com/{repository}/tar.gz/{commit}",
                 archive, item["archive_sha256"])
        with tempfile.TemporaryDirectory(dir=ROOT / "upstream") as temp:
            prefix = f"{name}-{commit}/"
            staging = Path(temp) / name
            staging.mkdir()
            with tarfile.open(archive) as stream:
                for member in stream.getmembers():
                    if member.isdir() and member.name == prefix.rstrip("/"):
                        continue
                    if not member.name.startswith(prefix):
                        raise ValueError("Unexpected archive root")
                    relative = Path(member.name[len(prefix):])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe archive path")
                    target = staging / relative
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = stream.extractfile(member)
                        with source, target.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
                        target.chmod(member.mode & 0o777)
                    else:
                        raise ValueError("Links and special archive entries are unsupported")
            staging.rename(destination)
        print(f"Restored {name} at {commit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()
    restore_sources()
    if args.data:
        info = json.loads((ROOT / "manifests/data.json").read_text())
        download(info["url"], ROOT / "data/pusht.zip", info["archive_sha256"])
        print("Data archive ready; install dependencies, then run scripts/prepare_data.py")


if __name__ == "__main__":
    main()
