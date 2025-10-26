from __future__ import annotations
import numpy as np
import torch, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from tetris.envs.tetris_env import TetrisEnvMetrics
from tetris.src.wrappers import CastObsToFloat32

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_env(app: str, persona: str, n_envs: int=8, frame_stack: int=1):
    def maker():
        if app == "tetris":
            return TetrisEnvMetrics(persona=persona, render_mode=None, seed=None)
        else:
            raise ValueError(f"Unknown app: {app}")
    env = make_vec_env(maker, n_envs=n_envs, wrapper_class=CastObsToFloat32)
    if frame_stack > 1 and app == "tetris":
        env = VecFrameStack(env, n_stack=frame_stack)
    return env
