import subprocess
import unittest
from scheduler_guard import require_terminal


class SchedulerGuardTest(unittest.TestCase):
    def runner(self, queue='', accounting='29427|TIMEOUT|\n', failure=False):
        def run(command, **kwargs):
            if failure:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0,
                queue if command[0] == 'squeue' else accounting, '')
        return run

    def test_retired_timed_out_job_is_allowed(self):
        self.assertEqual(require_terminal('29427', self.runner())['scheduler_state'], 'TIMEOUT')

    def test_active_job_is_rejected_even_with_old_terminal_record(self):
        with self.assertRaises(RuntimeError):
            require_terminal('29427', self.runner(queue='29427\n'))

    def test_missing_running_and_duplicate_accounting_are_rejected(self):
        for value in ['', '29427|RUNNING|\n', '29427|TIMEOUT|\n29427|TIMEOUT|\n']:
            with self.subTest(accounting=value), self.assertRaises(RuntimeError):
                require_terminal('29427', self.runner(accounting=value))

    def test_scheduler_failure_is_not_treated_as_completion(self):
        with self.assertRaises(subprocess.CalledProcessError):
            require_terminal('29427', self.runner(failure=True))


if __name__ == '__main__':
    unittest.main()
