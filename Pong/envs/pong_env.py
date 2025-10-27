# envs/pong_env.py
"""
Simple Gym-style Pong environment.

Observation: np.array([ball_x, ball_y, ball_vx, ball_vy, paddle_y], dtype=float32)
Actions: Discrete(3): 0=stay, 1=up, 2=down
Info exposes metrics: 'hits', 'misses', 'score', 'steps', 'event'
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, width=400, height=300, paddle_h=60, seed: int | None = None, max_steps=2000):
        super().__init__()
        self.width = width
        self.height = height
        self.paddle_h = paddle_h
        self.paddle_w = 10
        self.ball_size = 8
        self.paddle_speed = 6.0
        self.ball_speed = 4.0
        self.max_steps = max_steps

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -self.ball_speed, -self.ball_speed, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height, self.ball_speed, self.ball_speed, float(self.height)], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # RNG
        self._rng = np.random.default_rng(seed)
        self.seed_val = seed

        # State variables
        self.reset()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.seed_val = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # initialize state
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2
        self.ball_vx = self._rng.choice([-1, 1]) * self.ball_speed
        self.ball_vy = self._rng.uniform(-1, 1) * self.ball_speed
        self.paddle_y = (self.height - self.paddle_h) / 2
        self.hits = 0
        self.misses = 0
        self.score = 0
        self.steps = 0

        obs = self._get_obs()   # 5-element array
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_y], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        # (keep your existing step logic here)

        if self.steps >= self.max_steps:
            truncated = True  # time limit reached
            info['event'] = info.get('event', 'timeout')

        info.update({
            'hits': int(self.hits),
            'misses': int(self.misses),
            'score': int(self.score),
            'steps': int(self.steps)
        })

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode='human'):
        # text render for headless runs (keeps CI-friendly)
        print(f"[Pong] step={self.steps} ball=({self.ball_x:.1f},{self.ball_y:.1f}) paddle_y={self.paddle_y:.1f} hits={self.hits} misses={self.misses}")

    def close(self):
        pass
