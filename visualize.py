import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from tetris_gym import TetrisEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CastObsToFloat32(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        obs = observation.astype(np.float32, copy = False)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs

ALGOS = {"ppo": PPO, "a2c": A2C}

def make_env(render_mode:str):
    def _thunk():
        env = TetrisEnv(render_mode=render_mode)
        env = CastObsToFloat32(env)
        return env
    return _thunk


def main():
    p = argparse.ArgumentParser("Tetris Model Visualiser")
    p.add_argument("--model", required=True)
    p.add_argument("--algo", choices=ALGOS.keys(), default="ppo")
    p.add_argument("--device", default="cuda")
    p.add_argument("--frame_stack", type=int, default=4, help="Must match training (use 1 if no stacking)")
    p.add_argument("--render_mode", choices=["human", "rgb_array"], default="human")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()

    vec_env = DummyVecEnv([make_env(args.render_mode)])
    if args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, args.frame_stack)

    Model = ALGOS[args.algo]
    model = Model.load(args.model, device = args.device)

    obs = vec_env.reset()
    print("Watching model play Tetris...(Ctrl-C to quit)")

    # Run one episode with deterministic actions
    while True:
        action, _states = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = vec_env.step(action)
        try:
            vec_env.render()
        except Exception:
            pass
        if np.any(done):
            obs = vec_env.reset()


if __name__ == "__main__":
    main()