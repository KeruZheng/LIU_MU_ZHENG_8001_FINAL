import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from .geometry import normalize_xy, rotate_xy, rotate_image


def make_split(episode_count, seed=20261116, val_count=6):
    if not 0 < val_count < episode_count:
        raise ValueError("Need nonempty disjoint train and validation episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(episode_count).tolist()
    train, val = order[val_count:], order[:val_count]
    # Prefixes of this ordering stay nested and balanced within one trajectory.
    orientation = {str(e): i % 4 for i, e in enumerate(train)}
    orientation.update({str(e): i % 4 for i, e in enumerate(val)})
    return {"seed": seed, "episode_count": episode_count, "train": train,
            "validation": val, "orientation": orientation}


class PushTSequences(Dataset):
    def __init__(self, path, split, demos=50, horizon=16, obs_steps=2,
                 action_steps=8, rotated=False, cache=False):
        self.path = Path(path)
        self.root = zarr.open_group(str(path), mode="r")
        self.ends = np.asarray(self.root["meta/episode_ends"], dtype=np.int64)
        if len(self.ends) != split["episode_count"]:
            raise ValueError("Split does not match this dataset")
        if not 1 <= demos <= len(split["train"]):
            raise ValueError("Requested more independent training demos than available")
        self.ids = split["train"][:demos]
        self.orientation = split["orientation"]
        self.horizon, self.obs_steps, self.rotated = horizon, obs_steps, rotated
        self.indices = []
        for episode in self.ids:
            start = 0 if episode == 0 else int(self.ends[episode - 1])
            end = int(self.ends[episode])
            for offset in range(-(obs_steps - 1), end - start - horizon + action_steps):
                self.indices.append((episode, start, end, offset))
        if not self.indices:
            raise ValueError("No valid sequences")
        self.index_array = np.asarray(self.indices, dtype=np.int64)
        self.cache = None
        if cache:
            # Keep uint8 images in RAM; decode each zarr chunk only once per run.
            self.cache = {key: np.asarray(self.root[f"data/{key}"])
                          for key in ("img", "state", "action")}

    def get_batch(self, indices):
        if self.cache is None:
            rows = [self[int(i)] for i in indices]
            return {key: torch.stack([row[key] for row in rows])
                    for key in ("image", "agent_pos", "action")}
        episode, start, end, offset = self.index_array[np.asarray(indices)].T
        steps = np.clip(start[:, None] + offset[:, None] + np.arange(self.horizon),
                        start[:, None], end[:, None] - 1)
        obs = steps[:, :self.obs_steps]
        image = np.moveaxis(self.cache["img"][obs].astype(np.float32) / 255, -1, -3)
        agent = normalize_xy(self.cache["state"][obs, :2].astype(np.float32))
        action = normalize_xy(self.cache["action"][steps].astype(np.float32))
        if self.rotated:
            orientations = np.array([self.orientation[str(e)] for e in episode])
            for k in range(1, 4):
                mask = orientations == k
                image[mask] = rotate_image(image[mask], k)
                agent[mask] = rotate_xy(agent[mask], k)
                action[mask] = rotate_xy(action[mask], k)
        return {"image": torch.from_numpy(np.ascontiguousarray(image)),
                "agent_pos": torch.from_numpy(agent), "action": torch.from_numpy(action)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode, start, end, offset = self.indices[index]
        indices = np.clip(start + offset + np.arange(self.horizon), start, end - 1)
        obs_indices = indices[:self.obs_steps]
        image = np.asarray(self.root["data/img"].oindex[obs_indices]).astype(np.float32) / 255
        image = np.moveaxis(image, -1, -3)
        agent = np.asarray(self.root["data/state"].oindex[obs_indices])[:, :2].astype(np.float32)
        action = np.asarray(self.root["data/action"].oindex[indices]).astype(np.float32)
        agent, action = normalize_xy(agent), normalize_xy(action)
        if self.rotated:
            k = self.orientation[str(episode)]
            image, agent, action = rotate_image(image, k), rotate_xy(agent, k), rotate_xy(action, k)
        return {"image": torch.from_numpy(image.copy()), "agent_pos": torch.from_numpy(agent.copy()),
                "action": torch.from_numpy(action.copy()), "episode": episode}


def dataset_metadata(path):
    root = zarr.open_group(str(path), mode="r")
    ends = np.asarray(root["meta/episode_ends"])
    return {"episode_count": len(ends), "frames": int(ends[-1]),
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in root["data"].arrays()},
            "episode_ends_sha256": hashlib.sha256(ends.tobytes()).hexdigest()}
