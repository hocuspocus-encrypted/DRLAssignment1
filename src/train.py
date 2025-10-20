from __future__ import annotations
import argparse, os, sys
from stable_baselines3 import PPO, A2C
from tetris.src.utils import build_env, set_seed

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)

ALGOS = {
    'ppo': PPO,
    'a2c': A2C,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(ALGOS.keys()), default='ppo')
    parser.add_argument('--app', choices=['tetris'], default='tetris')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--persona', choices=('survivor', 'explorer'), default='survivor')
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--save', default='models/model')

    parser.add_argument('--a2c_n_steps', type=int, default=32)
    parser.add_argument('--a2c_lr', type=float, default=7e-4)
    parser.add_argument('--a2c_ent_coef', type=float, default=0.01)
    parser.add_argument('--a2c_vf_coef', type=float, default=0.5)
    parser.add_argument('--a2c_gamma', type=float, default=0.99)
    args = parser.parse_args()

    set_seed(args.seed)
    env = build_env(args.app, args.persona,n_envs=args.n_envs, frame_stack=args.frame_stack)

    policy = 'MlpPolicy'
    Model = ALGOS[args.algo]

    kwargs = dict(verbose=1, device=args.device, tensorboard_log=args.logdir)
    if args.algo == 'ppo':
        kwargs.update(n_steps=1024)
    else:
        kwargs.update(
            n_steps=args.a2c_n_steps,
            learning_rate=args.a2c_lr,
            ent_coef=args.a2c_ent_coef,
            vf_coef=args.a2c_vf_coef,
            gamma=args.a2c_gamma,
        )

    model = Model(policy, env, **kwargs)
    model.learn(total_timesteps=args.steps)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save(args.save)

if __name__ == '__main__':
    main()


