"""Build an E4-only overlay; all existing project environments stay read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    work = project / 'operations/e4_parallel_20261123'
    sources = work / 'sources'
    target = project / 'envs/e4_20261123'
    assert not target.exists(), 'Use a new environment version for a retry'
    provenance = json.loads((sources / 'provenance.json').read_text())
    provenance['pytorch3d'] = {'archive_sha256': json.loads((sources / 'pytorch3d_provenance.json').read_text())['sha256']}
    for name, item in provenance.items():
        archive = sources / (name + '.tar.gz')
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item['archive_sha256'], name
        dest = sources / name
        dest.mkdir(exist_ok=False)
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                assert not member.issym() and not member.islnk()
                assert (dest / member.name).resolve().is_relative_to(dest.resolve())
            stream.extractall(dest)
    subprocess.run([str(project / 'envs/robotic/bin/python'), '-m', 'venv', str(target)], check=True)
    site = target / 'lib/python3.10/site-packages'
    bases = [project / f'envs/{name}/lib/python3.10/site-packages' for name in ('official_dp', 'rq3', 'robotic')]
    (site / 'project_dependencies.pth').write_text('\n'.join(map(str, bases)) + '\n')
    env = dict(os.environ, PATH=str(target / 'bin') + ':' + os.environ['PATH'],
               PIP_CACHE_DIR=str(work / 'pip-cache'), TMPDIR=str(work / 'tmp'),
               PYTORCH3D_NO_EXTENSION='1', OPENBLAS_NUM_THREADS='1')
    (work / 'tmp').mkdir(exist_ok=True)
    pip = [str(target / 'bin/python'), '-m', 'pip', 'install', '--no-deps', '--ignore-installed']
    packages = ['mujoco==2.3.2', 'glfw==2.7.0', 'PyOpenGL==3.1.7',
        'h5py==3.10.0', 'dill==0.4.0', 'imagecodecs==2023.9.18',
        'imageio-ffmpeg==0.4.9', 'numba==0.59.1', 'llvmlite==0.42.0',
        'fvcore==0.1.5.post20221221', 'iopath==0.1.10', 'yacs==0.1.8',
        'portalocker==2.10.1', 'termcolor==2.4.0', 'gdown==5.2.0',
        'beautifulsoup4==4.12.3', 'soupsieve==2.6', 'PySocks==1.7.1',
        'chardet==5.2.0', 'tensorboardX==2.6.2.2', 'tensorboard==2.17.1',
        'opencv-python-headless==4.9.0.80']
    subprocess.run(pip + packages, check=True, env=env)
    for name in ('pytorch3d', 'robosuite', 'robomimic', 'robosuite-task-zoo', 'mimicgen'):
        src = next(p for p in (sources / name).iterdir() if p.is_dir())
        command = pip + ['--no-build-isolation']
        if name in ('robosuite-task-zoo', 'mimicgen'):
            command += ['-e']
        subprocess.run(command + [str(src)], check=True, env=env)
    result = subprocess.run([str(target / 'bin/python'), '-m', 'pip', 'freeze'],
                            check=True, capture_output=True, text=True, env=env)
    (work / 'environment.freeze.txt').write_text(result.stdout)
    (work / 'environment_ready.json').write_text(json.dumps({'state': 'installed',
        'python': str(target / 'bin/python'), 'read_only_bases': list(map(str, bases)),
        'gpu_preflight_required': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
