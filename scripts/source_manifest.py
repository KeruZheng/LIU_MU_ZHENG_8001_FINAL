"""Hash project source and pinned upstream source, excluding mutable outputs."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
for directory in ["src", "scripts", "tests", "configs", "slurm", "upstream/diffusion_policy", "upstream/equidiff"]:
    paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
                 and p.name != "clusters.local.json")
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}"
         for path in sorted(set(paths))]
(ROOT / "manifests/code.sha256").write_text("\n".join(lines) + "\n")
print(f"Recorded {len(lines)} source files")
