"""Install the Open3D import dependency required by pinned robomimic."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True, type=Path)
args = parser.parse_args()
project = args.project.resolve()
work = project / 'operations/e4_parallel_20261123'
python = project / 'envs/e4_20261123/bin/python'
constraints = work / 'open3d_constraints.txt'
constraints.write_text('numpy==1.26.4\nscipy==1.15.3\npillow==12.1.0\n')
env = dict(os.environ, PIP_CACHE_DIR=str(work / 'pip-cache'), TMPDIR=str(work / 'tmp'))
subprocess.run([str(python), '-m', 'pip', 'install', '--ignore-installed',
                '--constraint', str(constraints), 'open3d==0.18.0'], env=env, check=True)
result = subprocess.run([str(python), '-m', 'pip', 'freeze'], env=env,
                        capture_output=True, text=True, check=True)
(work / 'environment.freeze.txt').write_text(result.stdout)
