"""Inspect rendering libraries and test only the Slurm-allocated GPU."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

record = {'environment': {k: os.environ.get(k) for k in (
    'SLURM_JOB_ID', 'SLURMD_NODENAME', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES',
    'MUJOCO_EGL_DEVICE_ID', 'LD_LIBRARY_PATH', '__EGL_VENDOR_LIBRARY_FILENAMES')}}
for name, command in (
        ('gpu', ['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,driver_version', '--format=csv']),
        ('libraries', ['ldconfig', '-p'])):
    result = subprocess.run(command, text=True, capture_output=True)
    record[name] = result.stdout if name == 'gpu' else '\n'.join(
        line for line in result.stdout.splitlines() if any(x in line for x in ('EGL', 'GLX', 'nvidia')))
record['vendors'] = {str(p): p.read_text() for p in Path('/usr/share/glvnd/egl_vendor.d').glob('*.json')}
try:
    import torch
    x = torch.ones(4, device='cuda', requires_grad=True)
    x.square().sum().backward()
    record['cuda'] = {'name': torch.cuda.get_device_name(), 'torch': torch.__version__, 'backward_passed': True}
    from mujoco.egl import egl_ext as EGL
    devices = EGL.eglQueryDevicesEXT()
    record['egl_device_count'] = len(devices)
    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        EGL.eglGetProcAddress(b'eglQueryDeviceStringEXT'))
    record['egl_devices'] = [{'index': i, 'extensions': str(query(d, 0x3055)),
                             'drm': str(query(d, 0x3233))} for i, d in enumerate(devices)]
    from robosuite.renderers.context.egl_context import EGLGLContext
    context = EGLGLContext(84, 84)
    context.make_current()
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=84, width=84)
    renderer.update_scene(data)
    pixels = renderer.render()
    record['render'] = {'shape': list(pixels.shape), 'finite': bool(np.isfinite(pixels).all()),
                        'sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
    renderer.close()
    context.free()
    record['state'] = 'passed'
except Exception:
    record['state'] = 'failed'
    record['traceback'] = traceback.format_exc()
output = Path(os.environ['E4_DIAGNOSTIC_OUTPUT'])
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(0 if record['state'] == 'passed' else 1)
