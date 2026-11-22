"""Freeze the E4 protocol, upstream code and matched training configurations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--codes', type=Path, required=True)
    args = parser.parse_args()
    root = args.project / 'releases/e4_square_d0_local_20261122_v1'
    root.mkdir(exist_ok=False)
    for folder in ('results', 'logs', 'configs', 'runs', 'preflight'):
        (root / folder).mkdir()
    ops = Path(__file__).resolve().parent
    for name in ('run.py', 'supervise.py', 'prepare_data.py', 'data_sources.json', 'PROTOCOL.md'):
        shutil.copy2(ops / name, root / name)
    upstream = args.project / 'releases/stack_d1_s0_local_20261122_v1/upstream'
    source_manifest = json.loads((upstream.parent / 'frozen_files.json').read_text())
    for name, expected in source_manifest.items():
        assert hashlib.sha256((upstream.parent / name).read_bytes()).hexdigest() == expected, name
    shutil.copytree(upstream, root / 'upstream', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('get_max_steps', lambda name: 400, replace=True)
    OmegaConf.register_new_resolver('get_ws_x_center', lambda name: 0., replace=True)
    OmegaConf.register_new_resolver('get_ws_y_center', lambda name: 0., replace=True)
    for method, source in [('equidiff', 'train_equi_diffusion_unet_abs'), ('dp', 'train_diffusion_unet')]:
        for seed in (0, 1, 2):
            with initialize_config_dir(config_dir=str((root / 'upstream/equi_diffpo/config').resolve()), version_base=None):
                cfg = compose(config_name=source, overrides=['n_demo=100', 'task_name=square_d0'])
            OmegaConf.resolve(cfg)
            OmegaConf.set_struct(cfg, False)
            cfg.dataset_path = 'data/robomimic/datasets/square_d0/square_d0_first100_abs.hdf5'
            cfg.task.dataset.dataset_path = cfg.dataset_path
            cfg.task.env_runner.dataset_path = cfg.dataset_path
            cfg.training.seed = seed
            cfg.training.resume = False
            cfg.logging.mode = 'disabled'
            cfg.logging.name = f'e4_{method}_square_d0_s{seed}'
            for loader in (cfg.dataloader, cfg.val_dataloader):
                loader.batch_size = 64
                loader.num_workers = 2
            cfg.dataloader.drop_last = True
            cfg.task.env_runner.n_envs = 4
            OmegaConf.save(cfg, root / 'configs' / f'{method}_s{seed}.yaml')
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()}
    (root / 'frozen_files.json').write_text(json.dumps(files, indent=2) + '\n')
    print(root)


if __name__ == '__main__':
    main()
