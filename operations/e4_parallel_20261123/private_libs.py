"""Extract graphics/C++ runtime packages into the project, never install system packages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
args = parser.parse_args()
work = args.project.resolve() / 'operations/e4_parallel_20261123'
packages = work / 'runtime-debs'
target = work / 'runtime-libs'
packages.mkdir(exist_ok=True)
target.mkdir(exist_ok=True)
subprocess.run(['apt-get', 'download', 'libgl1', 'libglx0', 'libc++1-18',
                'libc++abi1-18', 'libunwind-18'], cwd=packages, check=True)
manifest = []
for package in sorted(packages.glob('*.deb')):
    subprocess.run(['dpkg-deb', '--extract', str(package), str(target)], check=True)
    manifest.append({'name': package.name, 'sha256': hashlib.sha256(package.read_bytes()).hexdigest()})
(work / 'runtime-libraries.json').write_text(json.dumps({'packages': manifest,
    'library_paths': [str(target / 'usr/lib/x86_64-linux-gnu'), str(target / 'usr/lib/llvm-18/lib')],
    'system_packages_installed': False}, indent=2) + '\n')
