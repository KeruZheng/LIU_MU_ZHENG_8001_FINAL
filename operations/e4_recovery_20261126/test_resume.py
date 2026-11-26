"""Small deterministic continuation test using the real optimizer and EMA."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE = WORKSPACE / 'development/proposal_v2/releases/e4_square_d0_local_20261122_v1'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'upstream'))
import dill
import torch
import equi_diffpo.model.diffusion.ema_model as ema_module
from resume_support import restore

OriginalEMA = ema_module.EMAModel


class Workspace:
    def __init__(self):
        torch.manual_seed(123)
        self.model = torch.nn.Linear(3, 2)
        self.ema_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.global_step = 0
        self.epoch = 0


def updates(workspace, ema, count):
    x = torch.tensor([[0.1, 0.2, 0.3]])
    for _ in range(count):
        workspace.optimizer.zero_grad()
        workspace.model(x).square().sum().backward()
        workspace.optimizer.step()
        ema.step(workspace.model)


class ResumeTest(unittest.TestCase):
    def test_continuation_matches_uninterrupted_weights_and_ema(self):
        full = Workspace()
        updates(full, OriginalEMA(full.ema_model), 8)
        partial = Workspace()
        updates(partial, OriginalEMA(partial.ema_model), 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt = root / 'periodic.ckpt'
            torch.save({'state_dicts': {k: getattr(partial, k).state_dict()
                                       for k in ['model', 'ema_model', 'optimizer']},
                        'pickles': {'epoch': dill.dumps(1), 'global_step': dill.dumps(3)}},
                       ckpt, pickle_module=dill)
            resumed = Workspace()
            restore(resumed, ckpt, 2, root / 'receipt.json')
            self.assertEqual((resumed.epoch, resumed.global_step), (2, 4))
            ema = ema_module.EMAModel(resumed.ema_model)
            self.assertEqual(ema.optimization_step, 4)
            updates(resumed, ema, 4)
            for name in ['model', 'ema_model']:
                for expected, actual in zip(getattr(full, name).parameters(), getattr(resumed, name).parameters()):
                    self.assertTrue(torch.equal(expected, actual), name)

    def test_rejects_partial_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt = root / 'partial.ckpt'
            torch.save({'pickles': {'epoch': dill.dumps(1), 'global_step': dill.dumps(2)}},
                       ckpt, pickle_module=dill)
            with self.assertRaises(AssertionError):
                restore(Workspace(), ckpt, 2, root / 'receipt.json')

    def tearDown(self):
        ema_module.EMAModel = OriginalEMA


if __name__ == '__main__':
    unittest.main()
