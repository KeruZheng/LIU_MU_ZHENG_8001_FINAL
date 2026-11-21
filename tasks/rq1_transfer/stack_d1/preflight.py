"""Exercise the pinned Stack-D1 image path without starting an experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3] / "upstream/equidiff"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("get_max_steps", lambda name: 400, replace=True)
    with initialize_config_dir(config_dir=str(root / "equi_diffpo/config"), version_base=None):
        return compose(config_name="train_equi_diffusion_unet_abs", overrides=["n_demo=100"])


def simulation(dataset, output):
    import h5py
    import numpy as np
    import robomimic.utils.file_utils as file_utils
    from equi_diffpo.env_runner.robomimic_image_runner import create_env
    from equi_diffpo.common.robomimic_util import RobomimicAbsoluteActionConverter
    from equi_diffpo.gym_util.video_recording_wrapper import VideoRecorder

    cfg = config()
    env = create_env(file_utils.get_env_metadata_from_dataset(str(dataset)), cfg.shape_meta)
    obs = env.reset()
    shapes = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        frame = obs[key]
        assert frame.shape == (3, 84, 84), (key, frame.shape)
        assert np.isfinite(frame).all() and frame.std() > 0
        shapes[key] = list(frame.shape)
    obs, reward, done, _ = env.step(np.zeros(7))
    frame = (obs["agentview_image"].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(frame).save(output / "agentview.png")
    video = VideoRecorder.create_h264(fps=10, thread_count=1)
    video.start(str(output / "render.mp4"))
    for _ in range(3):
        video.write_frame(frame)
    video.stop()
    env.env.close()

    # A separate two-demo file cannot be mistaken for the formal 100-demo input.
    smoke = output / "smoke_two_demos_abs.hdf5"
    if smoke.exists():
        raise FileExistsError(smoke)
    converter = RobomimicAbsoluteActionConverter(str(dataset))
    errors = []
    with h5py.File(dataset, "r") as src, h5py.File(smoke, "w") as dst:
        group = dst.create_group("data")
        for key, value in src["data"].attrs.items():
            group.attrs[key] = value
        total = 0
        for index in range(2):
            actions, info = converter.convert_and_eval_idx(index)
            assert np.isfinite(actions).all()
            src.copy(f"data/demo_{index}", group)
            group[f"demo_{index}/actions"][:] = actions
            total += len(actions)
            errors.append(info)
        group.attrs["total"] = total
    converter.file.close()
    converter.env.env.close()
    converter.abs_env.env.close()
    return {"camera_shapes": shapes, "step_reward": float(reward), "step_done": bool(done),
            "action_conversion_errors": errors, "smoke_dataset": str(smoke),
            "smoke_sha256": hashlib.sha256(smoke.read_bytes()).hexdigest()}


def model(dataset, output):
    import hydra
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from equi_diffpo.common.pytorch_util import dict_apply
    from equi_diffpo.model.common.rotation_transformer import RotationTransformer
    from equi_diffpo.workspace.train_equi_workspace import TrainEquiWorkspace

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    assert free > 10 * 1024**3, "Wait for at least 10 GiB free before this smoke check"
    torch.cuda.set_per_process_memory_fraction(0.25)
    rotation = RotationTransformer("axis_angle", "rotation_6d")
    rotation_errors = {}
    for device in ("cpu", "cuda"):
        value = torch.randn(8, 3, device=device) * 0.2
        error = (rotation.inverse(rotation.forward(value)) - value).abs().max().item()
        assert error < 1e-5
        rotation_errors[device] = error
    cfg = config()
    cfg.task.dataset.dataset_path = str(dataset)
    cfg.task.dataset.n_demo = 2
    cfg.task.dataset.use_cache = False
    ds = hydra.utils.instantiate(cfg.task.dataset)
    batch = dict_apply(next(iter(DataLoader(ds, batch_size=1, num_workers=0))),
                       lambda x: x.to("cuda"))
    workspace = TrainEquiWorkspace(cfg, output_dir=str(output))
    policy = workspace.model
    policy.set_normalizer(ds.get_normalizer())
    policy.to("cuda").train()
    started = time.monotonic()
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    # Upstream ModuleAttrMixin uses empty parameters only to track device/dtype.
    missing = [name for name, p in policy.named_parameters()
               if p.requires_grad and p.numel() and p.grad is None]
    print("Parameters without gradients:", missing, flush=True)
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert not missing, missing
    workspace.optimizer.step()
    workspace.ema_model.set_normalizer(ds.get_normalizer())
    workspace.ema_model.to("cuda")
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    ema.step(policy)
    policy.eval()
    with torch.no_grad():
        prediction = policy.predict_action(batch["obs"])
    assert prediction["action"].shape == (1, 8, 10)
    assert torch.isfinite(prediction["action"]).all()
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "batch_size": 1,
            "parameters": sum(p.numel() for p in policy.parameters()),
            "loss": loss.item(), "optimizer_step": True, "ema_step": True,
            "prediction_shape": list(prediction["action"].shape),
            "inference_steps": policy.num_inference_steps,
            "rotation_roundtrip_max_error": rotation_errors,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "elapsed_seconds": time.monotonic() - started,
            "formal_batch_128_validated": False}


def vector(dataset, output):
    import hydra
    import numpy as np

    cfg = config()
    runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=str(output), dataset_path=str(dataset),
        n_train=0, n_train_vis=0, n_test=1, n_test_vis=0, n_envs=1, max_steps=8)
    try:
        obs = runner.env.reset()
        action = np.zeros((1, 8, 7))
        action[:, :, :3] = obs["robot0_eef_pos"][:, -1:, :]
        obs, rewards, done, info = runner.env.step(action)
        assert np.isfinite(obs["agentview_image"]).all()
        return {"workers": 1, "steps": 8, "rewards": rewards.tolist(),
                "done": done.tolist(), "obs_shapes": {k: list(v.shape) for k, v in obs.items()}}
    finally:
        runner.env.close()


def workspace(dataset, output):
    import torch
    from equi_diffpo.workspace.train_equi_workspace import TrainEquiWorkspace

    torch.set_num_threads(2)
    assert torch.cuda.mem_get_info()[0] > 10 * 1024**3
    torch.cuda.set_per_process_memory_fraction(0.25)
    cfg = config()
    cfg.dataset_path = str(dataset)
    cfg.task.dataset.n_demo = 2
    cfg.task.dataset.use_cache = False
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 1
        loader.num_workers = 0
        loader.persistent_workers = False
    cfg.training.num_epochs = 1
    cfg.training.max_train_steps = 1
    cfg.training.max_val_steps = 1
    cfg.training.resume = False
    cfg.logging.mode = "disabled"
    cfg.checkpoint.topk.k = 0
    for key, value in dict(n_train=0, n_train_vis=0, n_test=1,
                           n_test_vis=0, n_envs=1, max_steps=8).items():
        cfg.task.env_runner[key] = value
    smoke = TrainEquiWorkspace(cfg, output_dir=str(output))
    smoke.run()
    if smoke._saving_thread is not None:
        smoke._saving_thread.join()
    checkpoint = smoke.get_checkpoint_path()
    assert checkpoint.is_file()
    # Deserialize the official checkpoint format on the new PyTorch version.
    import dill
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    assert {"model", "ema_model", "optimizer"} <= payload["state_dicts"].keys()
    return {"train_batches": 1, "validation_batches": 1,
            "rollout_episodes": 1, "rollout_max_steps": 8,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_deserialized": True,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "formal_batch_128_validated": False}


def data(dataset, output):
    import hydra
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(2)
    cfg = config()
    cfg.dataset_path = str(dataset)
    ds = hydra.utils.instantiate(cfg.task.dataset)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    normalizer = ds.get_normalizer()
    normalized_obs = normalizer.normalize(batch["obs"])
    normalized_actions = normalizer["action"].normalize(batch["action"])
    assert all(torch.isfinite(value).all() for value in normalized_obs.values())
    assert torch.isfinite(normalized_actions).all()
    cache = Path(str(dataset) + ".100..zarr.zip")
    assert cache.is_file()
    return {"recipe_n_demo": 100, "train_episodes": int(ds.train_mask.sum()),
            "validation_episodes": int((~ds.train_mask).sum()),
            "frames": int(ds.replay_buffer.n_steps), "train_sequences": len(ds),
            "action_shape": list(batch["action"].shape),
            "obs_shapes": {key: list(value.shape) for key, value in batch["obs"].items()},
            "normalizer_finite": True, "cache_bytes": cache.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("simulation", "model", "vector", "workspace", "data"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = globals()[args.stage](args.dataset.resolve(), args.output.resolve())
    result["scope"] = "engineering smoke check; not a trained policy or performance result"
    (args.output / f"{args.stage}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
