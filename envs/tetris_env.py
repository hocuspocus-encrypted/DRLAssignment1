from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

try:
    from DRLAssignment1.tetris_gym import TetrisEnv as BaseTetris
except Exception:
    BaseTetris = None

class TetrisEnvMetrics(BaseTetris):
    def __init__(self, persona: str = "survivor", render_mode: Optional[str] = None, seed: Optional[int] = None):
        assert BaseTetris is not None, "Please ensure tetris_gym.py with Tetris environment is available."
        super().__init__(render_mode=render_mode, seed=seed)
        self.persona = persona
        self._episode_metrics: Dict[str, Any] = {}
        self._reset_metrics()

    def _reset_metrics(self):
        self._episode_metrics = dict(
            steps=0,
            lines=0,
            hard_drops=0,
            soft_steps=0,
            left_moves=0,
            right_moves=0,
            rotations=0,
            deaths=0,
        )
    def reset(self, *,seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_metrics()
        info.update(self._episode_metrics)
        return obs, info

    def step(self, action: int):
        obs, r, term, trunc, info = super().step(action)
        if action == 1:self._episode_metrics["left_moves"] += 1
        if action == 2:self._episode_metrics["right_moves"] += 1
        if action == 3:self._episode_metrics["rotations"] += 1
        if action == 4:self._episode_metrics["soft_steps"] += 1
        if action == 5:self._episode_metrics["hard_drops"] += 1
        self._episode_metrics['steps'] += 1
        # Pull live counters from base env info
        self._episode_metrics['lines'] = info.get('cleared_lines', self._episode_metrics['lines'])

        if self.persona == 'survivor':
            r += 0.001 - 0.0002 * int(action in (1, 2))
        elif self.persona == 'explorer':
            r += 0.0005 * int(action in (1, 2, 3)) - 0.001 * int(action == 5)

        if term or trunc:
            self._episode_metrics['deaths'] += int(term)
            info.update(self._episode_metrics)
        else:
            info.update(self._episode_metrics)
        return obs, r, term, trunc, info
