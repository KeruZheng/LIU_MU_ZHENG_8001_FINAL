import unittest

from collect import classify, progress_alert, training_rate


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.steps = [16000, 18000, 20000]
        self.reports = {s: {'state': 'complete', 'summary': {'episodes': 100}} for s in self.steps}
        self.training = {'state': 'complete', 'step': 20000}

    def test_training_alone_does_not_mean_complete(self):
        self.assertNotEqual(classify('RUNNING', self.training, {}, self.steps, 'test'), 'complete')

    def test_partial_evaluation_not_complete(self):
        self.reports[20000]['summary']['episodes'] = 96
        self.assertNotEqual(classify('RUNNING', self.training, self.reports, self.steps, 'test'), 'complete')

    def test_scheduler_failure_overrides_stale_training(self):
        value = classify('NODE_FAIL', {'state': 'training'}, {}, self.steps, 'test')
        self.assertEqual(value, 'failed_node_fail')

    def test_finished_run_survives_later_queue_failure(self):
        self.assertEqual(classify('FAILED', self.training, self.reports, self.steps, 'test'), 'complete')

    def test_pending_is_not_computing(self):
        self.assertEqual(classify('PENDING', {'state': 'training'}, {}, self.steps, 'test'), 'pending')

    def test_unreachable_does_not_reuse_success_as_live(self):
        self.assertEqual(classify('UNKNOWN', self.training, self.reports, self.steps, 'test', False), 'unknown_stale_evidence')

    def test_stall_depends_on_progress_not_file_touch(self):
        previous = {'lane': {'token': 'step1', 'changed_at': 100}}
        self.assertTrue(progress_alert('lane', 'step1', previous, 2001, True)[1])
        self.assertFalse(progress_alert('lane', 'step2', previous, 2001, True)[1])
        self.assertFalse(progress_alert('lane', 'step1', previous, 2001, False)[1])

    def test_resume_throughput_uses_only_this_attempt(self):
        row = {'state': 'training', 'step': 21000, 'attempt_start_step': 20000, 'elapsed_seconds': 500}
        rate = training_rate(row, 40000)
        self.assertEqual(rate['updates_per_second'], 2)
        self.assertEqual(rate['remaining_training_seconds'], 9500)
        self.assertIsNone(training_rate({'state': 'complete', 'step': 20000}, 20000))


if __name__ == '__main__':
    unittest.main()
