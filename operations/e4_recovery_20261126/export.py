"""Export a completed local final EMA without optimizer/training tensors."""
import argparse
import json
from pathlib import Path
import shutil

from evaluate import atomic, digest

parser = argparse.ArgumentParser()
parser.add_argument('--release', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--method', choices=['equidiff', 'dp'], default='equidiff')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
run = args.release / 'runs' / f'{args.method}_square_d0_s{args.seed}'
completed = json.loads((run / 'completed.json').read_text())
assert completed['state'] == 'complete' and completed['epoch'] == 500
assert completed['global_step'] == 112000
checkpoint = run / 'checkpoints/final.ckpt'
args.output.mkdir(parents=True, exist_ok=False)
import dill
import h5py
import torch

payload = torch.load(checkpoint, map_location='cpu', weights_only=False, pickle_module=dill)
epoch = dill.loads(payload['pickles']['epoch'])
step = dill.loads(payload['pickles']['global_step'])
assert epoch == 500 and step == 112000
weights = args.output / 'final_ema.pt'
torch.save({'ema_model': payload['state_dicts']['ema_model'], 'epoch': epoch,
            'global_step': step}, weights)
config = args.output / 'config.yaml'
shutil.copyfile(args.release / f'configs/{args.method}_s{args.seed}.yaml', config)
proof = json.loads((args.release / 'results/data_ready.json').read_text())
metadata = args.output / 'metadata'
metadata.mkdir()
for domain in ('d0', 'd1', 'd2'):
    with h5py.File(metadata / f'{domain}.hdf5', 'w') as f:
        f.create_group('data').attrs['env_args'] = json.dumps(proof['sources']['square_' + domain]['env_meta'])
atomic(args.output / 'manifest.json', {'method': args.method, 'training_seed': args.seed,
    'epoch': epoch, 'global_step': step, 'source_checkpoint_sha256': digest(checkpoint),
    'weights_sha256': digest(weights), 'weights_bytes': weights.stat().st_size,
    'config_sha256': digest(config), 'metadata_sha256': {
        domain: digest(metadata / f'{domain}.hdf5') for domain in ('d0', 'd1', 'd2')},
    'source_release': args.release.name, 'selection': 'final epoch-500 EMA'})
print((args.output / 'manifest.json').read_text())
