# tetris_gym.py
# A compact Tetris environment compatible with Gymnasium / SB3
# Author: ChatGPT
# License: MIT

from __future__ import annotations
import math
import torch
from stable_baselines3 import PPO
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# Prefer Gymnasium; fall back to classic gym if needed
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces

try:
    import pygame  # optional (only needed for render_mode="human")
except Exception:  # pragma: no cover
    pygame = None


@dataclass
class Piece:
    id: int
    rotations: List[np.ndarray]  # list of (h, w) binary arrays


# Define Tetromino shapes in their rotation states (0/1 arrays)
# Using standard Tetris SRS minimal set of rotations (no kicks; simple collision checks)

def _rotations_from(shape: List[str]) -> List[np.ndarray]:
    arr = np.array([[1 if ch == 'X' else 0 for ch in row] for row in shape], dtype=np.uint8)
    rots = []
    a = arr.copy()
    for _ in range(4):
        if not any((np.array_equal(a, r) for r in rots)):
            rots.append(a.copy())
        a = np.rot90(a, k=-1)  # rotate clockwise
    return rots


PIECES: List[Piece] = [
    Piece(0, _rotations_from([
        "XXXX",
    ])),  # I
    Piece(1, _rotations_from([
        "XX",
        "XX",
    ])),  # O
    Piece(2, _rotations_from([
        " X ",
        "XXX",
    ])),  # T
    Piece(3, _rotations_from([
        " X",
        " X",
        "XX",
    ])),  # J
    Piece(4, _rotations_from([
        "X ",
        "X ",
        "XX",
    ])),  # L
    Piece(5, _rotations_from([
        " XX",
        "XX ",
    ])),  # S
    Piece(6, _rotations_from([
        "XX ",
        " XX",
    ])),  # Z
]


class TetrisEnv(gym.Env):
    """A lightweight Tetris environment for DRL.

    Observation (Box uint8): (2, H, W)
      - layer 0: settled board cells (0/1)
      - layer 1: current falling piece cells (0/1)

    Actions (Discrete 6):
      0: No-op (gravity tick)
      1: Move Left
      2: Move Right
      3: Rotate CW
      4: Soft Drop (one row)
      5: Hard Drop (lock)

    Reward shaping:
      - line clear: {1: +1.0, 2: +3.0, 3: +5.0, 4: +8.0}
      - soft drop: +0.01 per row moved by gravity or soft drop
      - hard drop: +0.02 per row moved
      - step penalty: -0.005 per step
      - game over: -1.0

    Episode terminates on game over.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self,
                 width: int = 10,
                 height: int = 20,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.W = int(width)
        self.H = int(height)
        self.render_mode = render_mode
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)
        # Board with settled blocks (H, W), uint8 {0,1}
        self.board = np.zeros((self.H, self.W), dtype=np.uint8)
        # Current piece state
        self.piece: Piece = PIECES[0]
        self.rot_idx: int = 0
        self.px: int = 0
        self.py: int = 0
        self.score: float = 0.0
        self.cleared_lines_total: int = 0

        # Gym spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.H, self.W, 2), dtype=np.uint8)

        # Rendering
        self._screen = None
        self._clock = None
        self.cell_size = 24

    # --- Gym API ---
    def seed(self, seed: Optional[int] = None):  # for classic gym
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self.board.fill(0)
        self.score = 0.0
        self.cleared_lines_total = 0
        self._spawn_new_piece()
        obs = self._get_obs()
        info = {"score": self.score, "cleared_lines": self.cleared_lines_total}
        return obs, info

    def step(self, action: int):
        reward = -0.005  # step penalty
        terminated = False
        truncated = False

        # Apply action
        if action == 1:
            self._try_move(dx=-1, dy=0)
        elif action == 2:
            self._try_move(dx=1, dy=0)
        elif action == 3:
            self._try_rotate()
        elif action == 4:
            moved = self._try_move(dx=0, dy=1)
            if moved:
                reward += 0.01
        elif action == 5:
            drop_rows = 0
            while self._try_move(dx=0, dy=1):
                drop_rows += 1
            reward += 0.02 * drop_rows
            self._lock_piece()
            lines = self._clear_lines()
            reward += self._line_reward(lines)
            self.cleared_lines_total += lines
            if not self._spawn_new_piece():
                reward -= 1.0
                terminated = True
        # Gravity tick if not hard-dropped
        if action != 5 and not terminated:
            if not self._try_move(dx=0, dy=1):
                self._lock_piece()
                lines = self._clear_lines()
                reward += self._line_reward(lines)
                self.cleared_lines_total += lines
                if not self._spawn_new_piece():
                    reward -= 1.0
                    terminated = True
            else:
                reward += 0.01  # gravity progress

        self.score += reward
        obs = self._get_obs()
        info = {"score": self.score, "cleared_lines": self.cleared_lines_total}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # --- Game mechanics ---
    def _spawn_new_piece(self) -> bool:
        self.piece = self._rng.choice(PIECES)
        self.rot_idx = 0
        shape = self._cur_shape()
        self.px = (self.W - shape.shape[1]) // 2
        self.py = 0
        # If spawn collides, it's game over
        return self._can_place(self.px, self.py, shape)

    def _cur_shape(self) -> np.ndarray:
        return self.piece.rotations[self.rot_idx]

    def _can_place(self, x: int, y: int, shape: np.ndarray) -> bool:
        h, w = shape.shape
        if x < 0 or y < 0 or x + w > self.W or y + h > self.H:
            return False
        region = self.board[y:y+h, x:x+w]
        return np.all((region + shape) <= 1)

    def _try_move(self, dx: int, dy: int) -> bool:
        nx, ny = self.px + dx, self.py + dy
        if self._can_place(nx, ny, self._cur_shape()):
            self.px, self.py = nx, ny
            return True
        return False

    def _try_rotate(self):
        next_idx = (self.rot_idx + 1) % len(self.piece.rotations)
        shape = self.piece.rotations[next_idx]
        # Basic wall kicks: try offsets
        for ox, oy in [(0,0), (-1,0), (1,0), (0,-1)]:
            if self._can_place(self.px + ox, self.py + oy, shape):
                self.rot_idx = next_idx
                self.px += ox
                self.py += oy
                return True
        return False

    def _lock_piece(self):
        shape = self._cur_shape()
        h, w = shape.shape
        self.board[self.py:self.py+h, self.px:self.px+w] |= shape

    def _clear_lines(self) -> int:
        full = np.all(self.board == 1, axis=1)
        lines = int(np.sum(full))
        if lines > 0:
            self.board = np.vstack([
                np.zeros((lines, self.W), dtype=np.uint8),
                self.board[~full]
            ])
        return lines

    def _line_reward(self, n: int) -> float:
        return {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}.get(n, 0.0)

    def _get_obs(self) -> np.ndarray:
        layer0 = self.board.copy()
        layer1 = np.zeros_like(self.board)
        shape = self._cur_shape()
        h, w = shape.shape
        y, x = self.py, self.px
        if 0 <= y < self.H and 0 <= x < self.W:
            y2 = min(y + h, self.H)
            x2 = min(x + w, self.W)
            sub_shape = shape[:y2 - y, :x2 - x]
            layer1[y:y2, x:x2] = np.maximum(layer1[y:y2, x:x2], sub_shape)
        obs = np.stack([layer0, layer1], axis=-1).astype(np.uint8)
        return obs

    # --- Rendering helpers ---
    def _render_rgb(self) -> np.ndarray:
        # Compose layers to RGB array
        scale = self.cell_size
        img = np.zeros((self.H * scale, self.W * scale, 3), dtype=np.uint8)
        # draw settled board
        for y in range(self.H):
            for x in range(self.W):
                if self.board[y, x]:
                    img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = (200, 200, 200)
        # draw current piece
        shp = self._cur_shape()
        h, w = shp.shape
        for dy in range(h):
            for dx in range(w):
                if shp[dy, dx]:
                    yy, xx = (self.py+dy)*scale, (self.px+dx)*scale
                    if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                        img[yy:yy+scale, xx:xx+scale] = (100, 180, 255)
        return img

    def _render_human(self):
        if pygame is None:
            raise RuntimeError("pygame is not installed. pip install pygame")
        if self._screen is None:
            pygame.init()
            w, h = self.W * self.cell_size, self.H * self.cell_size
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("TetrisEnv")
            self._clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(np.rot90(self._render_rgb()))
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata.get("render_fps", 12))


# -------- Convenience: manual play loop (optional) --------
if __name__ == "__main__":
    env = TetrisEnv(render_mode="human", seed=0)
    obs, info = env.reset()
    key_map = {
        pygame.K_LEFT: 1,
        pygame.K_RIGHT: 2,
        pygame.K_UP: 3,
        pygame.K_DOWN: 4,
        pygame.K_SPACE: 5,
    }
    done = False
    while True:
        a = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); raise SystemExit
            if event.type == pygame.KEYDOWN:
                a = key_map.get(event.key, 0)
        obs, r, term, trunc, info = env.step(a)
        env.render()
        if term or trunc:
            obs, info = env.reset()
