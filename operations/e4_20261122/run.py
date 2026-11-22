"""Matched-budget D0 training and untouched D1/D2 final-checkpoint tests."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from prepare_data import atomic, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--codes', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=['dp', 'equidiff'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--mode', required=True, choices=['preflight', 'train'])
    args = parser.parse_args()
    root = args.root.resolve()
    for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
        assert digest(root / name) == expected, name
    sys.path.insert(0, str(root / 'upstream'))
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from equi_diffpo.common.pytorch_util import dict_apply
    from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner

    assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(0.78)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = OmegaConf.load(root / 'configs' / f'{args.method}_s{args.seed}.yaml')
    assert cfg.n_demo == 100 and cfg.training.num_epochs == 500
    assert cfg.dataloader.batch_size == 64 and cfg.dataloader.drop_last
    assert not cfg.training.resume and not cfg.training.debug
    assert cfg.training.max_train_steps is None and cfg.training.max_val_steps is None
    data = args.codes / cfg.dataset_path
    receipt = json.loads(data.with_suffix('.conversion.json').read_text())
    assert digest(data) == receipt['output_sha256']
    cfg.dataset_path = str(data)
    cfg.task.dataset.dataset_path = str(data)
    cfg.task.env_runner.dataset_path = str(data)
    proof = json.loads((root / 'results/data_ready.json').read_text())
    target_paths = {'d0': data}
    for domain in ('d1', 'd2'):
        path = args.codes / f'data/robomimic/datasets/square_{domain}/square_{domain}_eval_metadata.hdf5'
        assert digest(path) == proof['sources']['square_' + domain]['metadata_sha256']
        target_paths[domain] = path
    run = f'{args.method}_square_d0_s{args.seed}'
    output = root / ('preflight' if args.mode == 'preflight' else 'runs') / run
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / 'effective_config.yaml')
    base = hydra.utils.get_class(cfg._target_)

    class AtomicWorkspace(base):
        def save_checkpoint(self, path=None, tag='latest', **kwargs):
            destination = Path(path) if path else self.get_checkpoint_path(tag)
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_suffix(destination.suffix + '.tmp')
            kwargs.pop('use_thread', None)
            super().save_checkpoint(path=tmp, use_thread=False, **kwargs)
            with tmp.open('rb') as stream:
                os.fsync(stream.fileno())
            tmp.replace(destination)
            return str(destination)

    workspace = AtomicWorkspace(cfg, output_dir=str(output))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batches = len(DataLoader(dataset, **cfg.dataloader))
    assert batches > 0
    status = {'state': 'preflight' if args.mode == 'preflight' else 'training', 'pid': os.getpid(),
              'method': args.method, 'seed': args.seed, 'target_epochs': 500,
              'batches_per_epoch': batches, 'target_updates': batches * 500,
              'batch_size': 64, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'started_at': time.time(), 'tf32': False,
              'training_input_sha256': receipt['output_sha256'],
              'train_episode_ids': np.flatnonzero(dataset.train_mask).tolist(),
              'validation_episode_ids': np.flatnonzero(~dataset.train_mask).tolist(),
              'selection': 'final epoch-500 EMA, fixed before any D1/D2 results'}
    atomic(output / 'runtime.json', status)

    def transfer(policy, domain, smoke=False):
        torch.manual_seed(9000000 + args.seed * 10 + int(domain[-1]))
        target = output / ('smoke_' + domain if smoke else 'transfer_' + domain)
        target.mkdir(exist_ok=False)
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(target),
            dataset_path=str(target_paths[domain]), n_train=0, n_train_vis=0,
            n_test=2 if smoke else 100, n_test_vis=0 if smoke else 2,
            test_start_seed=900000 if smoke else 200000,
            max_steps=8 if smoke else 400, n_envs=2 if smoke else 4)
        try:
            result = runner.run(policy)
        finally:
            runner.env.close()
        scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        episodes = [v for k, v in scores.items() if k.startswith('test/sim_max_reward_')]
        assert len(episodes) == (2 if smoke else 100)
        assert all(0 <= value <= 1 for value in episodes)
        atomic(target / 'scores.json', {'domain': domain, 'method': args.method, 'seed': args.seed,
            'checkpoint': 'engineering' if smoke else 'final_epoch_500_ema', 'episodes': len(episodes),
            'test_start_seed': 900000 if smoke else 200000, 'scores': scores})
        return scores

    if args.mode == 'preflight':
        normalizer = dataset.get_normalizer()
        for model in (workspace.model, workspace.ema_model):
            model.set_normalizer(normalizer)
            model.cuda()
        ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
        for i, batch in enumerate(DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)):
            batch = dict_apply(batch, lambda x: x.cuda())
            loss = workspace.model.compute_loss(batch)
            assert torch.isfinite(loss)
            loss.backward()
            workspace.optimizer.step()
            workspace.optimizer.zero_grad()
            ema.step(workspace.model)
            if i == 2:
                break
        workspace.ema_model.eval()
        with torch.no_grad():
            assert torch.isfinite(workspace.ema_model.predict_action(batch['obs'])['action']).all()
        for domain in target_paths:
            transfer(workspace.ema_model, domain, smoke=True)
        # Cover DP checkpoint serialization as well as EquiDiff's known path.
        checkpoint = Path(workspace.save_checkpoint())
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert 'state_dicts' in saved and 'optimizer' in saved['state_dicts']
        atomic(output / 'result.json', dict(status, state='complete', peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        return

    del dataset
    finished = threading.Event()
    def heartbeat():
        while not finished.is_set():
            atomic(output / 'status.json', dict(status, epoch=workspace.epoch,
                global_step=workspace.global_step, heartbeat=time.time()))
            finished.wait(10)
    original = RobomimicImageRunner.run
    def measured(runner, policy):
        old_state = status['state']
        status['state'] = 'evaluating'
        begin = time.monotonic()
        result = original(runner, policy)
        status.update(state=old_state, last_rollout_seconds=time.monotonic() - begin)
        return result
    RobomimicImageRunner.run = measured
    observer = threading.Thread(target=heartbeat, daemon=True)
    observer.start()
    try:
        workspace.run()
        assert workspace.epoch == 500 and workspace.global_step == batches * 500
        workspace.save_checkpoint(tag='final')
        workspace.ema_model.eval()
        status['state'] = 'transfer_evaluation'
        for domain in target_paths:
            transfer(workspace.ema_model, domain)
        inventory = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                     for p in sorted((output / 'checkpoints').glob('*.ckpt'))]
        atomic(output / 'local_inventory.json', {'state': 'retained_locally', 'files': inventory})
        status.update(state='complete', completed_at=time.time())
        atomic(output / 'completed.json', dict(status, epoch=workspace.epoch, global_step=workspace.global_step))
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        finished.set()
        observer.join()
        atomic(output / 'status.json', dict(status, epoch=workspace.epoch,
            global_step=workspace.global_step, heartbeat=time.time()))


if __name__ == '__main__':
    main()
