"""Wait for proven EGL recovery, then submit the reserved seed-2 lane when allowed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fleet-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-job', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / 'recovery.pid').write_text(str(os.getpid()) + '\n')
        config = json.loads(args.fleet_config.read_text())
        server = next(s for s in config['servers'] if s['name'] == 'sicc')
        work = server['root'] + '/operations/e4_parallel_20261123'
        started = time.time()

        def status(state, **kwargs):
            atomic(args.output / 'recovery_status.json', dict(state=state, heartbeat=time.time(),
                pid=os.getpid(), diagnostic_job=args.diagnostic_job, **kwargs))
            print(state, kwargs, flush=True)

        def rpc(command):
            return subprocess.run(server['ssh'] + [shlex.join(command)], capture_output=True,
                                  text=True, timeout=90)

        while time.time() - started < 48 * 3600:
            try:
                result = rpc(['sacct', '-j', args.diagnostic_job, '-X', '-nP', '--format=State,ExitCode'])
                result.check_returncode()
                scheduler = result.stdout.strip().split('|')[0]
                if scheduler in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'):
                    status('failed_diagnostic', scheduler=scheduler)
                    return
                if scheduler != 'COMPLETED':
                    status('waiting_diagnostic', scheduler=scheduler)
                else:
                    proof = rpc(['cat', work + '/egl_diagnostic_v2.json'])
                    proof.check_returncode()
                    assert json.loads(proof.stdout)['state'] == 'passed', 'Diagnostic did not prove rendering'
                    result = rpc(['python3', work + '/retry_egl.py', '--project', server['root'],
                        '--action', 'submit', '--diagnostic-job', args.diagnostic_job, '--account', 'qingbiaoli'])
                    if result.returncode:
                        if 'MaxSubmitJobsPerAccount' in result.stderr:
                            status('waiting_submission_slot', scheduler_error=result.stderr[-1500:])
                        else:
                            status('failed_submission', error=result.stderr[-2500:])
                            return
                    else:
                        receipt = rpc(['cat', work + '/submitted_s2_v2.json'])
                        receipt.check_returncode()
                        record = json.loads(receipt.stdout)
                        atomic(args.output / 'submitted_s2_v2.json', record)
                        # Update only ignored runtime configuration, never scientific source.
                        config = json.loads(args.fleet_config.read_text())
                        config['auxiliary'] = [e for e in config['auxiliary'] if e.get('release') != record['release']]
                        config['auxiliary'].append({'device': 'sicc', 'job_id': record['job_id'],
                            'name': 'E4 Square seed 2: private NVIDIA EGL recovery, both-model preflight and training',
                            'release': record['release'], 'owner': 'rq1_transfer'})
                        for entry in config['auxiliary']:
                            if entry.get('job_id') == '29342':
                                entry['resolved_by'] = 'Superseded by ' + record['job_id'] + '; new full-model preflight still required'
                        config['services'] = [e for e in config['services'] if e['name'] != 'E4 recovery dispatcher']
                        atomic(args.fleet_config, config)
                        status('submitted', job_id=record['job_id'], release=record['release'])
                        return
            except (subprocess.SubprocessError, OSError) as exc:
                status('retrying_connection', error=repr(exc))
            except Exception as exc:
                status('failed', error=repr(exc))
                return
            time.sleep(60)
        status('failed_timeout', error='Recovery dispatch did not finish within 48 hours')


if __name__ == '__main__':
    main()
