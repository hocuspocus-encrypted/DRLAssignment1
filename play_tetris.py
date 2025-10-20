# tetris_play.py
# Play/visualize TetrisEnv with keyboard controls using pygame.
# Requires: pip install pygame gymnasium numpy
# Usage: python tetris_play.py

import sys
import time
import numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit("pygame is required: pip install pygame") from e

from tetris_gym import TetrisEnv

WINDOW_SCALE = 28  # pixels per cell
MARGIN = 8  # outer padding
FPS = 30  # display refresh
GRAVITY_HZ = 6  # base gravity steps per second (when no soft drop)

# Key bindings
KEY_LEFT = pygame.K_LEFT
KEY_RIGHT = pygame.K_RIGHT
KEY_ROTATE = pygame.K_UP
KEY_SOFT = pygame.K_DOWN
KEY_HARD = pygame.K_SPACE
KEY_RESET = pygame.K_r
KEY_PAUSE = pygame.K_p
KEY_QUIT1 = pygame.K_ESCAPE
KEY_QUIT2 = pygame.K_q


class TetrisViewer:
    def __init__(self):
        self.env = TetrisEnv(render_mode="rgb_array", seed=0)
        obs, info = self.env.reset()
        self.info = info
        self.obs = obs
        self.paused = False

        self.h = self.env.H
        self.w = self.env.W

        pygame.init()
        self.clock = pygame.time.Clock()
        win_w = self.w * WINDOW_SCALE + 2 * MARGIN
        win_h = self.h * WINDOW_SCALE + 2 * MARGIN + 64  # space for HUD
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("TetrisEnv — Human Play")
        self.font = pygame.font.SysFont("consolas", 20)
        self.big = pygame.font.SysFont("consolas", 28, bold=True)

        # controls state (for key hold)
        self.hold_left = False
        self.hold_right = False
        self.hold_soft = False

        # key repeat timers (simple)
        self.repeat_delay = 0.16  # s before repeat kicks in
        self.repeat_rate = 0.05  # s between repeats
        self.last_left_time = 0.0
        self.last_right_time = 0.0

        self.last_gravity = time.time()
        self.gravity_interval = 1.0 / GRAVITY_HZ

    def step_env(self, action: int):
        self.obs, r, term, trunc, self.info = self.env.step(action)
        if term or trunc:
            # immediate reset so you can keep playing
            self.obs, self.info = self.env.reset()

    def draw(self):
        self.screen.fill((15, 15, 18))

        # Get RGB frame from env and blit
        frame = self.env._render_rgb()  # ndarray H*scale x W*scale x 3
        # We'll redraw ourselves with our own scale so use board info instead
        # Compose board + current piece from observation for sharper scaling
        # obs is (H,W,2) or (2,H,W); handle both
        if self.obs.ndim == 3 and self.obs.shape[-1] == 2:
            layer_board = self.obs[..., 0]
            layer_piece = self.obs[..., 1]
        elif self.obs.ndim == 3 and self.obs.shape[0] == 2:
            layer_board = self.obs[0]
            layer_piece = self.obs[1]
        else:
            # fallback: use env render
            surf = pygame.surfarray.make_surface(np.rot90(frame))
            self.screen.blit(surf, (MARGIN, MARGIN))

        # prefer cell-wise render from layers
        cell = WINDOW_SCALE
        top = MARGIN
        left = MARGIN
        for y in range(self.h):
            for x in range(self.w):
                rect = pygame.Rect(left + x * cell, top + y * cell, cell - 1, cell - 1)
                v = layer_board[y, x]
                p = layer_piece[y, x]
                if v:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                if p:
                    pygame.draw.rect(self.screen, (100, 180, 255), rect)
                # grid
                pygame.draw.rect(self.screen, (30, 30, 35), rect, 1)

        # HUD
        hud_y = MARGIN + self.h * cell + 8
        score_text = self.font.render(f"Score: {self.info.get('score', 0):.2f}", True, (230, 230, 240))
        lines_text = self.font.render(f"Lines: {self.info.get('cleared_lines', 0)}", True, (230, 230, 240))
        help_text = self.font.render("←/→ move  ↑ rotate  ↓ soft  SPACE hard  P pause  R reset  Q/Esc quit", True,
                                     (160, 160, 180))
        self.screen.blit(score_text, (MARGIN, hud_y))
        self.screen.blit(lines_text, (MARGIN + 220, hud_y))
        self.screen.blit(help_text, (MARGIN, hud_y + 26))

        if self.paused:
            overlay = self.big.render("PAUSED", True, (255, 220, 120))
            rect = overlay.get_rect(center=(self.screen.get_width() // 2, 28))
            self.screen.blit(overlay, rect)

        pygame.display.flip()

    def handle_inputs(self):
        now = time.time()
        action = 0  # default noop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in (KEY_QUIT1, KEY_QUIT2):
                    pygame.quit();
                    sys.exit(0)
                if event.key == KEY_PAUSE:
                    self.paused = not self.paused
                if event.key == KEY_RESET:
                    self.obs, self.info = self.env.reset()
                if event.key == KEY_ROTATE and not self.paused:
                    action = 3  # rotate
                if event.key == KEY_HARD and not self.paused:
                    action = 5  # hard drop
                if event.key == KEY_LEFT:
                    self.hold_left = True;
                    self.last_left_time = now - self.repeat_delay
                if event.key == KEY_RIGHT:
                    self.hold_right = True;
                    self.last_right_time = now - self.repeat_delay
                if event.key == KEY_SOFT:
                    self.hold_soft = True
            if event.type == pygame.KEYUP:
                if event.key == KEY_LEFT:
                    self.hold_left = False
                if event.key == KEY_RIGHT:
                    self.hold_right = False
                if event.key == KEY_SOFT:
                    self.hold_soft = False

        # held keys with repeat
        if not self.paused:
            if self.hold_left and (now - self.last_left_time) >= self.repeat_rate:
                action = 1;
                self.last_left_time = now
            if self.hold_right and (now - self.last_right_time) >= self.repeat_rate:
                action = 2;
                self.last_right_time = now
            if self.hold_soft:
                action = 4
        return action

    def run(self):
        while True:
            self.clock.tick(FPS)
            action = self.handle_inputs()

            # gravity tick based on timer; soft drop overrides
            if not self.paused:
                now = time.time()
                due_gravity = (now - self.last_gravity) >= self.gravity_interval
                if action == 0 and due_gravity:
                    # let env handle gravity by stepping no-op once
                    self.step_env(0)
                    self.last_gravity = now
                elif action != 0:
                    # apply immediate action
                    self.step_env(action)
                    # reset gravity timer slightly so piece has time before next tick
                    self.last_gravity = time.time()

            self.draw()


if __name__ == "__main__":
    TetrisViewer().run()
