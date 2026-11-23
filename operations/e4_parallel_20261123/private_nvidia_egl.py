"""Extract matching NVIDIA user-space graphics libraries; never install a driver."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True, type=Path)
args = parser.parse_args()
work = args.project.resolve() / 'operations/e4_parallel_20261123'
target = work / 'nvidia-egl-580.173.02'
downloads = work / 'nvidia-egl-debs'
target.mkdir(exist_ok=False)
downloads.mkdir(exist_ok=True)
inventory = []
for name in ('libnvidia-gl-580', 'libnvidia-gpucomp-580'):
    package = name + '=580.173.02-1ubuntu1'
    info = subprocess.check_output(['apt-cache', 'show', package], text=True)
    fields = dict(line.split(': ', 1) for line in info.splitlines() if ': ' in line and not line.startswith(' '))
    subprocess.run(['apt-get', 'download', package], cwd=downloads, check=True)
    path = downloads / Path(fields['Filename']).name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == fields['SHA256'], name
    subprocess.run(['dpkg-deb', '--extract', str(path), str(target)], check=True)
    inventory.append({'package': package, 'file': path.name, 'sha256': digest})
vendor = target / 'usr/share/glvnd/egl_vendor.d/10_nvidia.json'
assert vendor.exists()
record = {'driver_version': '580.173.02', 'packages': inventory,
          'library_path': str(target / 'usr/lib/x86_64-linux-gnu'),
          'vendor_file': str(vendor), 'system_driver_modified': False}
(work / 'nvidia-egl-runtime.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
