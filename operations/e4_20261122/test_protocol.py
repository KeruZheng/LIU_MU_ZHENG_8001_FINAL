"""Prevent target leakage and unmatched budgets in frozen E4 configs."""
import json
from pathlib import Path
import unittest
from omegaconf import OmegaConf


class ProtocolTests(unittest.TestCase):
    def test_training_uses_only_d0_and_equal_budgets(self):
        codes = Path(__file__).resolve().parents[2]
        root = codes.parent / 'development/proposal_v2/releases/e4_square_d0_local_20261122_v1'
        for seed in (0, 1, 2):
            configs = [OmegaConf.load(root / 'configs' / f'{method}_s{seed}.yaml') for method in ('dp', 'equidiff')]
            for cfg in configs:
                for path in (cfg.dataset_path, cfg.task.dataset.dataset_path, cfg.task.env_runner.dataset_path):
                    self.assertIn('square_d0_first100_abs', path)
                self.assertEqual(cfg.training.seed, seed)
                self.assertFalse(cfg.training.resume)
                self.assertFalse(cfg.training.debug)
                self.assertIsNone(cfg.training.max_train_steps)
            for field in ('n_demo', 'training.num_epochs', 'dataloader.batch_size', 'dataloader.drop_last',
                          'task.dataset.seed', 'task.dataset.val_ratio', 'policy.num_inference_steps'):
                self.assertEqual(OmegaConf.select(configs[0], field), OmegaConf.select(configs[1], field), field)

    def test_all_domains_are_pinned_and_distinct(self):
        sources = json.loads(Path(__file__).with_name('data_sources.json').read_text())
        self.assertEqual({Path(x['path']).stem for x in sources['files']}, {'square_d0', 'square_d1', 'square_d2'})
        self.assertEqual(len({x['lfs']['oid'] for x in sources['files']}), 3)
        self.assertEqual(len(sources['revision']), 40)


if __name__ == '__main__':
    unittest.main()
