"""Create fresh E4 lanes and gate their submission on a graphics diagnostic job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare', 'submit'], required=True)
    parser.add_argument('--diagnostic-job')
    parser.add_argument('--account')
    args = parser.parse_args()
    project = args.project.resolve()
    work = project / 'operations/e4_parallel_20261123'
    graphics = json.loads((work / 'nvidia-egl-runtime.json').read_text())
    libraries = json.loads((work / 'runtime-libraries.json').read_text())['library_paths']
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join([graphics['library_path']] + libraries),
               __EGL_VENDOR_LIBRARY_FILENAMES=graphics['vendor_file'])
    with (work / 'retry_v2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in (1, 2):
            old = project / f'releases/e4_square_d0_s{seed}_sicc_20261123_v1'
            root = old.with_name(old.name[:-2] + 'v2')
            if args.action == 'prepare':
                assert not (old / 'runs').exists() or not list((old / 'runs').iterdir()), 'Formal work already exists'
                root.mkdir(exist_ok=False)
                for name, expected in json.loads((old / 'frozen_files.json').read_text()).items():
                    assert sha(old / name) == expected, name
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old / name, target)
                for name in ('runs', 'results', 'logs', 'preflight'):
                    (root / name).mkdir(exist_ok=True)
                write(root / 'graphics_runtime.json', graphics)
                with (root / 'PROTOCOL.md').open('a') as stream:
                    stream.write('\n## EGL environment recovery\n\n'
                        'The v1 preflight reached CUDA training and sampling but failed to create an EGL display. '
                        'Allocated-GPU diagnosis confirmed working CUDA and only Mesa EGL vendor libraries. '
                        'This fresh attempt exports project-private NVIDIA EGL libraries matching driver 580.173.02. '
                        'No model, scientific configuration, simulator source or system driver changes. '
                        'The original failed attempt is retained. A separate CUDA/EGL diagnostic must succeed '
                        'before scheduling this lane; both full-model preflights are then repeated.\n')
                frozen = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
                write(root / 'frozen_files.json', frozen)
                registry = project / 'operations/parallel_20261119/registry.json'
                value = json.loads(registry.read_text())
                value['releases'] = sorted(set(value['releases']) | {root.name})
                write(registry, value)
                print('Prepared', root.name, flush=True)
            else:
                assert args.diagnostic_job and args.diagnostic_job.isdigit() and args.account
                receipt = work / f'submitted_s{seed}_v2.json'
                if receipt.exists():
                    prior = json.loads(receipt.read_text())
                    assert prior['release'] == root.name and prior['diagnostic_job'] == args.diagnostic_job
                    print('Already submitted', json.dumps(prior), flush=True)
                    continue
                for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
                    assert sha(root / name) == expected, name
                active = subprocess.check_output(['squeue', '-h', '--name', f'robotic_e4_s{seed}_v2',
                    '-u', str(os.getuid()), '-o', '%i'], text=True).strip()
                assert not active, f'Job exists without a receipt: {active}; reconcile before retrying'
                command = ['sbatch', '--parsable', '--account', args.account,
                    '--partition=a100_batch,h800_batch', '--time=3-00:00:00', '--export=ALL',
                    '--dependency=afterok:' + args.diagnostic_job, '--kill-on-invalid-dep=yes',
                    '--job-name', f'robotic_e4_s{seed}_v2', '--chdir', str(root),
                    '--output', str(root / 'logs/slurm_%j.out'), str(root / 'train.sbatch'), str(project), root.name]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                if result.returncode:
                    write(work / f'submit_error_s{seed}_v2.json', {'time': time.time(),
                        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
                    raise RuntimeError(result.stderr)
                job = result.stdout.strip().split(';')[0]
                assert job.isdigit(), result.stdout
                value = {'job_id': job, 'seed': seed, 'release': root.name, 'time': time.time(),
                    'diagnostic_job': args.diagnostic_job, 'command': command,
                    'graphics_environment': {k: env[k] for k in ('LD_LIBRARY_PATH', '__EGL_VENDOR_LIBRARY_FILENAMES')}}
                write(receipt, value)
                print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
