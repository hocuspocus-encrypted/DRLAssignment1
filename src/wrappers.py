try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces
import numpy as np

class CastObsToFloat32(gym.ObservationWrapper):
    """Cast HWC observations to float32 in [0,1] so SB3's CnnPolicy accepts them."""
    def __init__(self, env):
        super().__init__(env)
        low  = 0.0
        high = 1.0
        self.observation_space = spaces.Box(
            low=low, high=high, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        # obs may be uint8 0/1 or 0..255; this safely maps to float32
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        # If your env ever returns 0/255, divide once here:
        if self.observation_space.high.max() == 1.0 and obs.max() > 1.0:
            obs = obs / 255.0
        return obs
