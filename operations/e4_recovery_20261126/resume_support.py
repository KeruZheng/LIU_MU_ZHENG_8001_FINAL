"""Restore epoch-boundary training state without resetting EMA warmup."""
import json
from pathlib import Path


def restore(workspace, checkpoint, batches, receipt_path):
    import dill
    import torch
    from prepare_data import atomic, digest
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, pickle_module=dill)
    epoch = dill.loads(payload['pickles']['epoch'])
    step = dill.loads(payload['pickles']['global_step'])
    # Upstream periodic checkpoints are saved before the end-of-epoch increments.
    assert 0 <= epoch < 500 and step + 1 == (epoch + 1) * batches, (epoch, step, batches)
    assert set(payload['state_dicts']) == {'model', 'ema_model', 'optimizer'}
    discarded = {}
    for name in ('model', 'ema_model'):
        model = getattr(workspace, name)
        model.eval()
        eval_keys = set(model.state_dict())
        model.train()
        train_keys = set(model.state_dict())
        cache_keys = eval_keys - train_keys
        weights = payload['state_dicts'][name]
        discarded[name] = sorted(set(weights) & cache_keys)
        weights = {key: value for key, value in weights.items() if key not in cache_keys}
        model.load_state_dict(weights, strict=True)
        loaded = model.state_dict()
        assert all(torch.equal(loaded[k].cpu(), v.cpu()) for k, v in weights.items())
    optimizer = payload['state_dicts']['optimizer']
    assert optimizer['state'], 'Optimizer state is required; weights-only resume is forbidden'
    steps = {int(v['step']) for v in optimizer['state'].values() if 'step' in v}
    assert steps == {step + 1}, steps
    workspace.optimizer.load_state_dict(optimizer)
    workspace.epoch = epoch + 1
    workspace.global_step = step + 1
    # The upstream workspace creates EMA inside run(); retain its existing
    # weights while restoring the decay schedule to the completed update count.
    import equi_diffpo.model.diffusion.ema_model as ema_module
    original = ema_module.EMAModel

    class ContinuedEMA(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.optimization_step = workspace.global_step

    ema_module.EMAModel = ContinuedEMA
    receipt = {'checkpoint': str(checkpoint), 'sha256': digest(checkpoint),
               'saved_epoch': epoch, 'saved_step': step,
               'next_epoch': workspace.epoch, 'completed_updates': workspace.global_step,
               'optimizer_steps': sorted(steps), 'all_loaded_tensors_equal': True,
               'discarded_regenerated_eval_caches': discarded,
               'ema_optimization_step': workspace.global_step,
               'rng_state': 'Not saved by original checkpoint; seeded restart, not bitwise continuation',
               'scheduler': 'Original 500-epoch cosine horizon, last_epoch=completed_updates-1'}
    atomic(receipt_path, receipt)
    del payload
    return receipt
