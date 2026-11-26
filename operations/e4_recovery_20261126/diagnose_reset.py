"""Record reset differences after interaction without weakening any assertion."""
import argparse
import json
from pathlib import Path
import random
import sys

p = argparse.ArgumentParser()
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.upstream.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e4_corrected_eval_20261124'))
from evaluate import atomic, scene_record
import numpy as np
from omegaconf import OmegaConf
from equi_diffpo.env_runner.robomimic_image_runner import create_env
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as fu

cfg = OmegaConf.load(a.inputs / 'config.yaml')
meta = fu.get_env_metadata_from_dataset(str(a.inputs / 'metadata/d2.hdf5'))
meta['env_kwargs']['use_object_obs'] = False
meta['env_kwargs']['controller_configs']['control_delta'] = False
env = create_env(meta, cfg.shape_meta)
wrapper = RobomimicImageWrapper(env, cfg.shape_meta)
records = []
saved = None
try:
    for index, seed in enumerate([200014, 200015, 200014]):
        np.random.seed(seed)
        random.seed(seed)
        env.env.hard_reset = True
        obs = wrapper.get_observation(env.reset())
        record = scene_record(wrapper, obs, seed)
        if seed == 200014 and saved is not None:
            record['differences'] = {k: {'before': saved[0][k], 'after': v}
                                     for k, v in record.items() if saved[0][k] != v}
            record['obs_max_absolute_difference'] = {
                k: float(np.max(np.abs(v - saved[1][k]))) for k, v in obs.items()}
        if saved is None:
            saved = (record, {k: v.copy() for k, v in obs.items()})
        records.append(record)
        atomic(a.output / 'records.json', records)
        if index == 0:
            for _ in range(400):
                env.step(np.zeros(7))
    print(json.dumps(records[-1], indent=2), flush=True)
finally:
    env.env.close()
