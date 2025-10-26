from __future__ import annotations
import argparse, sys,os
from stable_baselines3 import PPO, A2C
from tetris.src.utils import build_env, set_seed
from DRLAssignment1.src.metrics import MetricsLogger

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)

ALGOS = {
    'ppo': PPO,
    'a2c': A2C
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(ALGOS.keys()), default='ppo')
    parser.add_argument('--app', choices=['tetris'], default='tetris')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--persona', choices=['survivor', 'explorer'], default='survivor')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outdir', default='logs/eval')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    set_seed(args.seed)
    env = build_env(args.app, args.persona, n_envs=1, frame_stack=args.frame_stack)

    Model = ALGOS[args.algo]
    model = Model.load(args.model, device=args.device)

    logger = MetricsLogger(args.outdir, run_name=f"{args.app}-{args.algo}-{args.persona}-{args.seed}")

    for episode in range(args.episodes+1):
        obs = env.reset()
        episode_reward, episode_length = 0.0,0
        last_action = {}
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            last_action = info[0] if isinstance(info, list) else info
            if done.any() if hasattr(done, 'any') else done:
                break

        extras = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'lines' : last_action.get('lines', ''),
            'hard_drops' : last_action.get('hard_drops', ''),
            'rotations' : last_action.get('rotations', ''),
            'left_moves' : last_action.get('left_moves', ''),
            'right_moves' : last_action.get('right_moves', ''),
            'soft_drops' : last_action.get('soft_drops', ''),
        }
        logger.add(episode, episode_reward, episode_length, **extras)
    logger.flush()

if __name__ == '__main__':
    main()