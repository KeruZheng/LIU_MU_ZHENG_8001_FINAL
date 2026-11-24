"""Independent E4 final-EMA evaluation with seeded full-scene resets.

Each four-episode shard is a fresh process, bounding EGL/model reset memory.
The upstream runner and policies are read-only; patching is process-local.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def scene_record(wrapper, obs, seed):
    import numpy as np
    sim = wrapper.env.env.sim
    model = sim.model
    peg = model.body_name2id('peg1')
    nut = model.body_name2id('SquareNut_main')
    arrays = {name: np.asarray(getattr(model, name)) for name in
              ('body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'cam_pos', 'cam_quat')}
    arrays['state'] = np.asarray(wrapper.env.get_state()['states'])
    hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    return {'seed': int(seed), 'peg_pos': sim.data.body_xpos[peg].tolist(),
            'peg_quat': sim.data.body_xquat[peg].tolist(),
            'nut_pos': sim.data.body_xpos[nut].tolist(),
            'state_and_model_hashes': hashes,
            'observation_hashes': {key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                   for key, value in obs.items()}}


def install_reset(audit_dir):
    import numpy as np
    from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

    def reset(self):
        assert self.init_state is None, 'Corrected evaluator supports seeded tests only'
        seed = self._seed if self._seed is not None else getattr(self, '_evaluation_seed', None)
        assert seed is not None, 'Every evaluation scene needs an explicit seed'
        self._evaluation_seed = int(seed)
        np.random.seed(seed)
        self.env.env.hard_reset = True
        # Never restore states-only caches: peg geometry lives in the model XML.
        raw_obs = self.env.reset()
        self._seed = None
        obs = self.get_observation(raw_obs)
        record = scene_record(self, obs, seed)
        path = audit_dir / f'{seed}.json'
        if path.exists():
            assert json.loads(path.read_text()) == record, f'Non-reproducible scene {seed}'
        else:
            atomic(path, record)
        return obs

    RobomimicImageWrapper.reset = reset
    return RobomimicImageWrapper


def imports(args):
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    assert torch.cuda.is_available(), 'GPU evaluation must run in a Slurm GPU allocation'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def audit(args):
    torch = imports(args)
    import numpy as np
    from omegaconf import OmegaConf
    from equi_diffpo.env_runner.robomimic_image_runner import create_env
    import robomimic.utils.file_utils as FileUtils
    wrapper_type = install_reset(args.output / 'scenes')
    cfg = OmegaConf.load(args.config)
    metadata = FileUtils.get_env_metadata_from_dataset(str(args.metadata))
    metadata['env_kwargs']['use_object_obs'] = False
    metadata['env_kwargs']['controller_configs']['control_delta'] = False
    np.random.seed(args.constructor_seed)
    env = create_env(metadata, cfg.shape_meta)
    wrapper = wrapper_type(env, cfg.shape_meta)
    rss = []
    try:
        for seed in (200000, 200001, 200002, 200000):
            wrapper.seed(seed)
            obs = wrapper.reset()
            assert all(np.isfinite(value).all() for value in obs.values())
            assert obs['agentview_image'].shape == (3, 84, 84)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        scenes = [json.loads((args.output / 'scenes' / f'{seed}.json').read_text())
                  for seed in range(200000, 200003)]
        assert len({tuple(s['nut_pos']) for s in scenes}) == 3
        if args.domain != 'd0':
            assert len({tuple(s['peg_pos']) for s in scenes}) == 3
        if args.domain == 'd2':
            assert len({tuple(s['peg_quat']) for s in scenes}) == 3
        atomic(args.output / 'audit.json', {'state': 'passed', 'domain': args.domain,
            'constructor_seed': args.constructor_seed, 'scenes': scenes, 'peak_rss_kib': rss,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'absolute_action': True})
    finally:
        env.env.close()


def shard(args):
    torch = imports(args)
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    install_reset(args.output / 'scenes')
    import equi_diffpo.env_runner.robomimic_image_runner as runner_module
    original_vector = runner_module.AsyncVectorEnv

    def audited_factory(factory):
        def create():
            # The pinned vector runner uses spawn, so patch inside each worker.
            install_reset(args.output / 'scenes')
            return factory()
        return create

    def audited_vector(env_fns, **kwargs):
        return original_vector([audited_factory(factory) for factory in env_fns], **kwargs)

    runner_module.AsyncVectorEnv = audited_vector
    payload = torch.load(args.weights, map_location='cpu', weights_only=False)
    assert payload['epoch'] == 500 and payload['global_step'] == 112000
    policy = hydra.utils.instantiate(cfg.policy)
    # ESCNN caches expanded filters in eval mode; the saved EMA contains the
    # train-mode parameters only. Remove caches, load strictly, then rebuild.
    policy.train()
    policy.load_state_dict(payload['ema_model'], strict=True)
    del payload
    policy.to('cuda').eval()
    # Fixed per-shard diffusion RNG; scene seeds are independent of this stream.
    torch.manual_seed(9000000 + args.training_seed * 10 + int(args.domain[-1]) + args.start)
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(args.output),
        dataset_path=str(args.metadata), n_train=0, n_train_vis=0, n_test=args.count,
        n_test_vis=0, test_start_seed=args.start, max_steps=args.steps, n_envs=args.count)
    started = time.time()
    try:
        result = runner.run(policy)
    finally:
        runner.env.close()
    scores = {str(seed): float(result[f'test/sim_max_reward_{seed}'])
              for seed in range(args.start, args.start + args.count)}
    assert all(value in (0.0, 1.0) for value in scores.values())
    atomic(args.output / 'scores.json', {'domain': args.domain, 'scores': scores,
        'max_steps': args.steps, 'seconds': time.time() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda})


def run_child(args, mode, domain, output, extra):
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', __file__, mode, '--upstream', str(args.upstream),
        '--config', str(args.config), '--weights', str(args.weights), '--output', str(output),
        '--metadata', str(args.metadata_dir / f'{domain}.hdf5'), '--domain', domain,
        '--training-seed', str(args.training_seed)] + extra
    with (output / 'run.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    with (args.output / 'suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((args.weights.parent / 'manifest.json').read_text())
        assert digest(args.weights) == manifest['weights_sha256']
        assert digest(args.config) == manifest['config_sha256']
        for domain in args.domains:
            assert digest(args.metadata_dir / f'{domain}.hdf5') == manifest['metadata_sha256'][domain]
        state = {'state': 'preflight', 'started_at': time.time(), 'domains': args.domains,
                 'training_seed': args.training_seed, 'weights_sha256': manifest['weights_sha256'],
                 'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
                 'script_sha256': digest(__file__), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                 'episodes_completed': 0, 'protocol': 'seeded_full_scene_v1',
                 'selection': 'final epoch-500 EMA; no target-based checkpoint selection'}
        try:
            for domain in args.domains:
                state.update(domain=domain, state='preflight', heartbeat=time.time())
                atomic(args.output / 'status.json', state)
                for repeat, constructor_seed in enumerate((12345, 67890)):
                    run_child(args, 'audit', domain, args.output / f'audit_{domain}_{repeat}',
                              ['--constructor-seed', str(constructor_seed)])
                records = [json.loads((args.output / f'audit_{domain}_{i}/audit.json').read_text())
                           for i in (0, 1)]
                assert records[0]['scenes'] == records[1]['scenes'], 'Constructor-dependent scenes'
                run_child(args, 'shard', domain, args.output / f'smoke_{domain}',
                          ['--start', '900000', '--count', '2', '--steps', '8'])
            atomic(args.output / 'preflight_passed.json', dict(state, state='passed', time=time.time()))
            for domain in args.domains:
                scores = {}
                for start in range(200000, 200100, 4):
                    state.update(state='evaluating', domain=domain, shard_start=start, heartbeat=time.time())
                    atomic(args.output / 'status.json', state)
                    path = args.output / domain / str(start)
                    run_child(args, 'shard', domain, path,
                              ['--start', str(start), '--count', '4', '--steps', '400'])
                    result = json.loads((path / 'scores.json').read_text())
                    scores.update(result['scores'])
                    state['episodes_completed'] += 4
                    state.update(heartbeat=time.time(), last_shard_seconds=result['seconds'])
                    atomic(args.output / 'status.json', state)
                assert set(scores) == {str(seed) for seed in range(200000, 200100)}
                scenes = [json.loads((args.output / domain / str(seed - seed % 4) / 'scenes' /
                                     f'{seed}.json').read_text()) for seed in range(200000, 200100)]
                if domain != 'd0':
                    assert len({tuple(item['peg_pos']) for item in scenes}) == 100
                atomic(args.output / domain / 'scores.json', {'domain': domain,
                    'successes': sum(scores.values()), 'episodes': 100,
                    'success_rate': sum(scores.values()) / 100, 'scores': scores,
                    'protocol': state['protocol'], 'weights_sha256': state['weights_sha256'],
                    'source_checkpoint_sha256': state['source_checkpoint_sha256'], 'scenes': scenes})
            state.update(state='complete', completed_at=time.time())
            atomic(args.output / 'completed.json', state)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), heartbeat=time.time())
            raise
        finally:
            atomic(args.output / 'status.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['suite', 'audit', 'shard'])
    for name in ('upstream', 'config', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--domains', nargs='+', choices=['d0', 'd1', 'd2'], default=['d1', 'd2', 'd0'])
    parser.add_argument('--domain', choices=['d0', 'd1', 'd2'])
    parser.add_argument('--training-seed', type=int, default=0)
    parser.add_argument('--constructor-seed', type=int, default=12345)
    parser.add_argument('--start', type=int, default=200000)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--steps', type=int, default=400)
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
