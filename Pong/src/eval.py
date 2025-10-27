# src/eval.py
"""
Evaluate a saved model on Pong and log per-episode metrics.
Usage:
 python src/eval.py --model models/pong_ppo_survivor_seed7.zip --algo ppo --persona survivor --seed 7 --episodes 50 --outdir logs/eval
"""
import argparse, os
from stable_baselines3 import PPO, A2C
from src.utils import build_env, set_seed, load_yaml, RewardWrapper
from src.metrics import MetricsLogger

ALGOS = {'ppo': PPO, 'a2c': A2C}

def evaluate(model_path, algo, persona_yaml, seed, episodes, outdir, device):
    persona_cfg = load_yaml(persona_yaml) if persona_yaml else {}
    set_seed(seed)
    env = build_env('pong', persona_cfg, seed=seed)
    env = RewardWrapper(env, persona_cfg)
    ModelClass = ALGOS[algo]
    model = ModelClass.load(model_path, device=device)
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(outdir)
    logger = MetricsLogger(outdir, run_name=f"pong-eval-{algo}-{os.path.splitext(os.path.basename(persona_yaml))[0]}-seed{seed}")

    for ep in range(episodes):
        obs = env.reset()
        ep_r, ep_len = 0.0, 0
        last_info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_r += reward
            ep_len += 1
            last_info = info if isinstance(info, dict) else {}
            if done:
                break
        extras = {
            'hits': last_info.get('hits',''),
            'misses': last_info.get('misses',''),
            'score': last_info.get('score',''),
            'steps': last_info.get('steps',''),
        }
        logger.add(ep, ep_r, ep_len, app='pong', algo=algo, persona=os.path.splitext(os.path.basename(persona_yaml))[0], seed=seed, extras=extras)
    csv, summary = logger.flush()
    print(f"Evaluation logs: {csv}, summary: {summary}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--algo', choices=['ppo','a2c'], default='ppo')
    parser.add_argument('--persona', choices=['survivor','explorer'], default='survivor')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--outdir', default='logs/eval')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    persona_yaml = os.path.join('configs','rewards', f"{args.persona}.yaml")
    evaluate(args.model, args.algo, persona_yaml, args.seed, args.episodes, args.outdir, args.device)
