"""Verify pinned Square inputs and convert only the 100 D0 training demos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', required=True, type=Path)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    records = {}
    for row in sources['files']:
        name = Path(row['path']).stem
        directory = args.codes / 'data/robomimic/datasets' / name
        raw = directory / (name + '.hdf5')
        candidate = raw if raw.exists() else raw.with_suffix('.hdf5.part')
        assert candidate.stat().st_size == row['size'], name + ' incomplete download'
        assert digest(candidate) == row['lfs']['oid'], name + ' checksum mismatch'
        if candidate != raw:
            candidate.rename(raw)
        with h5py.File(raw, 'r') as dataset:
            meta = json.loads(dataset['data'].attrs['env_args'])
            assert meta['env_name'] == 'Square_' + name[-2:].upper()
            records[name] = {'sha256': row['lfs']['oid'], 'bytes': row['size'],
                             'demos': len(dataset['data']), 'env_meta': meta}
        # Transfer evaluation needs metadata only, never target demonstrations.
        if name != 'square_d0':
            target = directory / (name + '_eval_metadata.hdf5')
            with h5py.File(target, 'x') as out:
                out.create_group('data').attrs['env_args'] = json.dumps(meta)
            records[name]['metadata_sha256'] = digest(target)
    directory = args.codes / 'data/robomimic/datasets/square_d0'
    source = directory / 'square_d0.hdf5'
    subset = directory / 'square_d0_first100.hdf5'
    with h5py.File(source, 'r') as src, h5py.File(subset, 'x') as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        group = dst.create_group('data')
        for k, v in src['data'].attrs.items():
            group.attrs[k] = v
        for i in range(100):
            src.copy('data/demo_' + str(i), group)
        group.attrs['total'] = sum(len(group[f'demo_{i}/actions']) for i in range(100))
    partial = directory / 'square_d0_first100_abs.hdf5.part'
    converted = directory / 'square_d0_first100_abs.hdf5'
    converter = args.codes / 'upstream/equidiff/equi_diffpo/scripts/robomimic_dataset_conversion.py'
    subprocess.run([sys.executable, str(converter), '-i', str(subset), '-o', str(partial), '-n', '2'], check=True)
    arrays = 0
    with h5py.File(subset, 'r') as src, h5py.File(partial, 'r') as dst:
        names, other = [], []
        src.visit(names.append)
        dst.visit(other.append)
        assert names == other
        for name in ['/'] + names:
            a, b = src[name], dst[name]
            assert set(a.attrs) == set(b.attrs)
            for key in a.attrs:
                assert np.array_equal(a.attrs[key], b.attrs[key]), (name, key)
            if not isinstance(a, h5py.Dataset):
                continue
            assert a.shape == b.shape and a.dtype == b.dtype
            if name.endswith('/actions'):
                assert np.isfinite(b[:]).all()
                assert np.array_equal(a[:, -1], b[:, -1])
            else:
                assert np.array_equal(a[:], b[:]), name
            arrays += 1
    # Prove the selected observations/actions were not changed during subsetting.
    with h5py.File(source, 'r') as src, h5py.File(subset, 'r') as dst:
        for i in range(100):
            group = dst[f'data/demo_{i}']
            def check(name, item):
                if isinstance(item, h5py.Dataset):
                    assert np.array_equal(item[:], src[f'data/demo_{i}/' + name][:])
            group.visititems(check)
    converted_hash = digest(partial)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    partial.rename(converted)
    receipt = {'state': 'verified', 'source_sha256': records['square_d0']['sha256'],
               'subset_sha256': digest(subset), 'output_sha256': converted_hash,
               'demos': 100, 'selection': 'demo_0 through demo_99, matching upstream n_demo=100',
               'verified_arrays': arrays, 'converter_sha256': digest(converter),
               'target_demonstrations_used_for_training': False}
    atomic(converted.with_suffix('.conversion.json'), receipt)
    atomic(args.output, {'state': 'complete', 'sources': records, 'conversion': receipt})


if __name__ == '__main__':
    main()
