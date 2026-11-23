"""Installation-only import check; does not create models, simulations or CUDA tensors."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.insert(0, str(args.root / 'upstream'))
modules = ['torch', 'torchvision', 'escnn', 'mujoco', 'robosuite', 'robomimic',
           'mimicgen_envs', 'robosuite_task_zoo', 'open3d', 'pytorch3d.transforms',
           'equi_diffpo.workspace.train_equi_workspace',
           'equi_diffpo.workspace.train_diffusion_unet_hybrid_workspace',
           'equi_diffpo.env_runner.robomimic_image_runner']
paths = {}
for name in modules:
    module = importlib.import_module(name)
    paths[name] = module.__file__
import torch
versions = {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'escnn', 'mujoco', 'robomimic', 'robosuite',
             'open3d', 'numpy', 'scipy', 'numba', 'gym', 'diffusers', 'imagecodecs')}
args.output.write_text(json.dumps({'state': 'imports_passed', 'versions': versions,
    'module_paths': paths, 'cuda_build': torch.version.cuda,
    'gpu_preflight_passed': False}, indent=2) + '\n')
print(json.dumps(versions))
