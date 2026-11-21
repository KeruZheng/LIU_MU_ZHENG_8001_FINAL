"""Run a frozen, full-recipe Stack-D1 reference on the local workstation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("preflight", "train"))
    args = parser.parse_args()
    root = args.root.resolve()
    for relative, expected in json.loads((root / "frozen_files.json").read_text()).items():
        assert digest(root / relative) == expected, relative
    sys.path.insert(0, str(root / "upstream"))

    import hydra
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from equi_diffpo.common.pytorch_util import dict_apply
    from equi_diffpo.workspace.train_equi_workspace import TrainEquiWorkspace

    assert torch.cuda.is_available()
    assert "5090" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.80)
    cfg = OmegaConf.load(root / "config.yaml")
    assert cfg.n_demo == 100 and cfg.training.num_epochs == 500
    assert cfg.dataloader.batch_size == 128 and cfg.policy.down_dims == [512, 1024, 2048]
    assert cfg.training.max_train_steps is None and cfg.training.max_val_steps is None
    assert not cfg.training.debug and not cfg.training.resume
    assert cfg.task.env_runner.n_test == 50 and cfg.task.env_runner.max_steps == 400
    receipt = json.loads(Path(cfg.dataset_path).with_suffix(".conversion.json").read_text())
    assert digest(Path(cfg.dataset_path)) == receipt["output_sha256"]

    output = root / ("preflight" if args.mode == "preflight" else "runs/stack_d1_n100_s0")
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "effective_config.yaml")

    class AtomicWorkspace(TrainEquiWorkspace):
        def save_checkpoint(self, path=None, tag="latest", **kwargs):
            destination = Path(path) if path else self.get_checkpoint_path(tag)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            kwargs.pop("use_thread", None)
            super().save_checkpoint(path=temporary, use_thread=False, **kwargs)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
            return str(destination)

    workspace = AtomicWorkspace(cfg, output_dir=str(output))
    if args.mode == "preflight":
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
        workspace.model.set_normalizer(dataset.get_normalizer())
        workspace.ema_model.set_normalizer(dataset.get_normalizer())
        workspace.model.to("cuda")
        workspace.ema_model.to("cuda")
        ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
        times = []
        for index, batch in enumerate(loader):
            batch = dict_apply(batch, lambda x: x.to("cuda"))
            torch.cuda.synchronize()
            start = time.monotonic()
            loss = workspace.model.compute_loss(batch)
            assert torch.isfinite(loss)
            loss.backward()
            workspace.optimizer.step()
            workspace.optimizer.zero_grad()
            ema.step(workspace.model)
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            print(json.dumps({"preflight_update": index + 1, "loss": loss.item(), "seconds": times[-1]}), flush=True)
            if index == 9:
                break
        workspace.ema_model.eval()
        with torch.no_grad():
            predictions = workspace.ema_model.predict_action(batch["obs"])
        assert torch.isfinite(predictions["action"]).all()
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output),
                                        n_train=0, n_train_vis=0, n_test=4, n_test_vis=0,
                                        max_steps=8)
        try:
            smoke_rollout = runner.run(workspace.ema_model)
        finally:
            runner.env.close()
        result = {"state": "complete", "batch_size": 128, "updates": len(times),
                  "steady_seconds_per_update": sum(times[2:]) / len(times[2:]),
                  "update_seconds": times, "sampling_batch_size": 128,
                  "rollout_workers": cfg.task.env_runner.n_envs,
                  "smoke_rollout": smoke_rollout,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                  "scope": "Resource preflight only, no performance interpretation"}
        atomic(output / "result.json", result)
        print(json.dumps(result), flush=True)
        return

    status = {"state": "training", "pid": os.getpid(), "seed": 0,
              "target_epochs": 500, "target_updates": 38500, "batch_size": 128,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "started_at": time.time(),
              "scope": "Official Stack-D1 image reference; separate from Push-T matrix"}
    atomic(output / "started.json", status)
    atomic(output / "runtime.json", status)
    finished = threading.Event()
    status_lock = threading.Lock()

    def heartbeat():
        while not finished.is_set():
            with status_lock:
                current = dict(status, heartbeat=time.time(), epoch=workspace.epoch,
                               global_step=workspace.global_step)
            atomic(output / "status.json", current)
            finished.wait(10)

    from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner
    original_run = RobomimicImageRunner.run

    def measured_run(runner, policy):
        begin = time.monotonic()
        with status_lock:
            status["state"] = "evaluating"
        result = original_run(runner, policy)
        with status_lock:
            status.update(state="training", last_rollout_seconds=time.monotonic() - begin,
                          last_test_mean_score=result.get("test/mean_score"))
        return result

    RobomimicImageRunner.run = measured_run
    observer = threading.Thread(target=heartbeat, daemon=True)
    observer.start()
    try:
        workspace.run()
        assert workspace.epoch == 500
        workspace.save_checkpoint(tag="final")
        workspace.ema_model.eval()
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output))
        try:
            final_result = runner.run(workspace.ema_model)
        finally:
            runner.env.close()
        atomic(output / "final_evaluation.json", {k: v for k, v in final_result.items()
                                                   if isinstance(v, (int, float))})
        inventory = []
        for checkpoint in sorted((output / "checkpoints").glob("*.ckpt")):
            inventory.append({"name": checkpoint.name, "bytes": checkpoint.stat().st_size,
                              "sha256": digest(checkpoint)})
        atomic(output / "local_inventory.json", {"state": "retained_locally", "files": inventory,
                                                   "scope": "Local originals, no deletion or second backup claimed"})
        status.update(state="complete", completed_at=time.time())
        atomic(output / "completed.json", dict(status, epoch=workspace.epoch,
                                                 global_step=workspace.global_step))
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        finished.set()
        observer.join()
        atomic(output / "status.json", dict(status, heartbeat=time.time(), epoch=workspace.epoch,
                                              global_step=workspace.global_step))


if __name__ == "__main__":
    main()
