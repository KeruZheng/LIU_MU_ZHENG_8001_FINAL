"""Native Push-T prototype. Physical and pixel symmetry need separate validation."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
from .geometry import WORLD_CENTER, rotate_state


class PushTRotEnv(PushTImageEnv):
    def __init__(self, orientation=0, **kwargs):
        self.orientation = int(orientation) % 4
        super().__init__(**kwargs)

    def _setup(self):
        super()._setup()
        self.goal_pose = np.array([WORLD_CENTER, WORLD_CENTER,
                                   np.pi / 4 - self.orientation * np.pi / 2])

    def reset(self):
        # Sample once in the canonical frame, then rotate the complete state.
        rng = np.random.RandomState(self._seed)
        canonical = np.array([rng.uniform(55.5, 455.5), rng.uniform(55.5, 455.5),
                              rng.uniform(105.5, 405.5), rng.uniform(105.5, 405.5),
                              rng.uniform(-np.pi, np.pi)])
        self.reset_to_state = rotate_state(canonical, self.orientation)
        return super().reset()


def coverage(env):
    goal = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    block = pymunk_to_shapely(env.block, env.block.shapes)
    return float(goal.intersection(block).area / goal.area)
