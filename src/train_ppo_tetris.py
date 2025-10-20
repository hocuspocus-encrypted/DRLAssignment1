# train_ppo_tetris.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# Import your env class
from tetris.tetris_gym import TetrisEnv

# Wrap in a callable for SB3
def make_env():
    return TetrisEnv(render_mode=None, seed=0)

env = make_vec_env(make_env, n_envs=8)  # vectorize for throughput
env = VecFrameStack(env, n_stack=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_tetris", device=device)
model.learn(total_timesteps=1_000_000)
model.save("ppo_tetris")

# Watch one episode
eval_env = TetrisEnv(render_mode="human", seed=1)
obs, info = eval_env.reset()
done = False
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = eval_env.step(int(action))
    eval_env.render()
    if term or trunc:
        obs, info = eval_env.reset()
