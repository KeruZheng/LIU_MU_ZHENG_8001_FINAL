"""Push-T study components; upstream sources remain unmodified."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream" / "diffusion_policy"
if UPSTREAM.is_dir():
    sys.path.insert(0, str(UPSTREAM))
