"""Watch the collector; only a dead collector may be restarted, never training."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from collect import atomic, check_process, read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--once', action='store_true')
    p.add_argument('--restart-collector', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (a.output / 'watchdog.pid').write_text(str(os.getpid()) + '\n')
        last_restart = 0
        while True:
            try:
                pid = (a.output / 'collector.pid').read_text().strip()
            except OSError:
                pid = None
            now = time.time()
            alive = check_process(pid, str(Path(__file__).with_name('collect.py')))
            snapshot = read(a.output / 'latest.json', {})
            age = now - snapshot.get('time_epoch', 0)
            status = {'time_epoch': now, 'collector_alive': alive, 'snapshot_age_seconds': age,
                      'errors': [], 'action': 'observe'}
            if not alive or age > 600:
                status['errors'].append('Collector absent or no successful snapshot for 10 minutes')
            if not alive and a.restart_collector and now - last_restart > 300:
                with (a.output / 'collector.log').open('a') as stream:
                    proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('collect.py')),
                        '--config', str(a.config), '--output', str(a.output)],
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                last_restart = now
                status.update(action='restart_collector', new_pid=proc.pid)
            atomic(a.output / 'watchdog.json', status)
            if a.once:
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
