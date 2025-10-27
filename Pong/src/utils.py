# src/utils.py
import os
import yaml
import numpy as np
import gymnasium as gym

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    # Seed numpy and python random if needed, and optionally other libs
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def build_env(env_name: str, persona: str = 'survivor'):
    """
    Build and return a gymnasium environment, optionally with reward shaping
    based on persona ('survivor' or 'explorer').
    """

    # Try to build Pong from various gym versions
    if env_name.lower() in ('pong', 'atari-pong'):
        tried = ['ALE/Pong-v5', 'PongNoFrameskip-v4', 'Pong-v0']
        for name in tried:
            try:
                env = gym.make(name, render_mode=None)
                print(f"✅ Using Atari env: {name}")
                break
            except Exception:
                env = None
                continue
        if env is None:
            print("⚠️ Atari Pong not available — falling back to CartPole-v1 for testing.")
            env = gym.make("CartPole-v1")
    else:
        env = gym.make(env_name)

    # Apply reward shaping wrapper
    env = PersonaRewardWrapper(env, persona)
    return env


class PersonaRewardWrapper(gym.RewardWrapper):
    """
    Reward wrapper for different personas:
      - Survivor: rewards staying alive / avoiding losing
      - Explorer: rewards making progress / hitting the ball
    """
    def __init__(self, env, persona):
        super().__init__(env)
        self.persona = persona.lower()

    def reward(self, reward):
        # For Pong: agent gets +1 for scoring, -1 for losing a point
        # For other envs (like CartPole), we adapt generically
        if self.persona == 'survivor':
            # Encourage staying alive / avoiding penalties
            shaped_reward = reward
            if reward < 0:
                shaped_reward *= 2  # penalize losing more heavily
            else:
                shaped_reward *= 0.5  # smaller reward for scoring
        elif self.persona == 'explorer':
            # Encourage active play / taking action
            shaped_reward = reward * 2.0
        else:
            shaped_reward = reward
        return shaped_reward
class RewardWrapper(gym.Wrapper):
    """
    Wrap an environment to shape rewards according to persona config.
    persona_cfg should be a dict with keys like:
      - reward.hit, reward.miss, reward.alive (floats)
    The wrapper intercepts step() call, inspects info,event fields, and returns shaped reward.
    Also returns the raw env.info under 'raw_info' key for logging.
    """
    def __init__(self, env, persona_cfg: dict):
        self.env = env
        self.persona = persona_cfg.get('reward', {}) if persona_cfg else {}
        # defaults
        self.r_hit = float(self.persona.get('hit', 1.0))
        self.r_miss = float(self.persona.get('miss', -5.0))
        self.r_alive = float(self.persona.get('alive', 0.01))
    def compute_custom_reward(self, obs, reward, done, info):
        """
        You can customize this function to shape rewards based on events or observations.
        For now, it just returns the original reward.
        """
        # Example: Give a small bonus for every successful hit
        if info.get("event") == "hit":
            reward += 0.5
        elif info.get("event") == "score":
            reward += 1.0
        elif info.get("event") == "miss":
            reward -= 1.0

        return reward

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated  # combine for compatibility

        # Modify reward if needed
        reward = self.compute_custom_reward(obs, reward, done, info)

        return obs, reward, terminated, truncated, info
    def render(self, mode='human'):
        return self.env.render(mode)
    def close(self):
        return self.env.close()
