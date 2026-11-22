"""Native C4 physics with a canonical-frame rasterizer aligned to the demos.

The original (256, 256) goal is rotated about (255.5, 255.5), including its
position. This deliberate subpixel protocol correction preserves the official
canonical images. It differs from pinning all goal positions to (256, 256).
"""

import numpy as np
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from .geometry import WORLD_CENTER, rotate_image, rotate_xy


def copy_body(source, target, k):
    angle = -int(k) * np.pi / 2
    target.angle = source.angle + angle
    target.position = tuple(rotate_xy(np.asarray(source.position), k, WORLD_CENTER))
    target.velocity = tuple(rotate_xy(np.asarray(source.velocity), k))
    target.force = tuple(rotate_xy(np.asarray(source.force), k))
    target.angular_velocity = source.angular_velocity
    target.torque = source.torque


class PushTC4Env(PushTImageEnv):
    def __init__(self, orientation=0, recenter=True, **kwargs):
        self.orientation = int(orientation) % 4
        self.recenter = recenter
        self.canonical_state = None
        self.raster = PushTImageEnv(**kwargs)
        super().__init__(**kwargs)

    def _setup(self):
        super()._setup()
        self.goal_pose[:2] = rotate_xy(self.goal_pose[:2], self.orientation, WORLD_CENTER)
        self.goal_pose[2] -= self.orientation * np.pi / 2

    def reset(self):
        self.raster.seed(self._seed)
        if self.canonical_state is not None:
            self.raster.reset_to_state = np.array(self.canonical_state, copy=True)
        elif self.recenter:
            rng = np.random.RandomState(self._seed)
            # Original integer sampler means 249.5, so shift by exactly six pixels.
            self.raster.reset_to_state = np.array([
                rng.randint(50, 450) + 6, rng.randint(50, 450) + 6,
                rng.randint(100, 400) + 6, rng.randint(100, 400) + 6,
                rng.randn() * 2 * np.pi - np.pi])
        else:
            self.raster.reset_to_state = None
        self.raster.reset()
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        copy_body(self.raster.agent, self.agent, self.orientation)
        copy_body(self.raster.block, self.block, self.orientation)
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)
        self.latest_action = None
        return self._get_obs()

    def _get_obs(self):
        # Rasterize inverse-transformed native state; rotate the final pixels losslessly.
        # The shadow space is never stepped here and cannot affect native dynamics.
        copy_body(self.agent, self.raster.agent, -self.orientation)
        copy_body(self.block, self.raster.block, -self.orientation)
        self.raster.space.reindex_shapes_for_body(self.raster.agent)
        self.raster.space.reindex_shapes_for_body(self.raster.block)
        self.raster.latest_action = None
        canonical = self.raster._get_obs()
        image = rotate_image(canonical["image"], self.orientation)
        self.render_cache = np.moveaxis((image * 255).astype(np.uint8), 0, -1)
        return {"image": image, "agent_pos": np.array(self.agent.position)}

    def close(self):
        self.raster.close()
        super().close()
