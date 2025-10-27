from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np

# Use relative imports since we're running as a module
try:
    from .utils import build_env, set_seed
except ImportError:
    from utils import build_env, set_seed

ALGOS = {
    'ppo': PPO,
    'a2c': A2C,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DRL agent on Pong.")
    parser.add_argument('--algo', choices=list(ALGOS.keys()), required=True, help='Algorithm used')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--env', default='pong', help='Environment name')
    parser.add_argument('--persona', choices=['survivor', 'explorer'], default='survivor', help='Reward persona type')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions')
    parser.add_argument('--save-results', type=str, default=None, help='Path to save evaluation results JSON')
    return parser.parse_args()

def make_eval_env(env_name, persona, seed, render=False):
    """Create a single evaluation environment"""
    def _init():
        try:
            env = build_env(env_name, persona=persona)
        except TypeError:
            print(f"Warning: build_env doesn't accept persona parameter")
            env = build_env(env_name)
        
        if render:
            try:
                env = env.unwrapped
                env.render_mode = 'human'
            except:
                print("Warning: Could not set render mode")
        
        env.reset(seed=seed)
        return env
    
    env = DummyVecEnv([_init])
    
    # Check if we need to transpose for CNN
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[-1] in [1, 3, 4]:
        env = VecTransposeImage(env)
    
    return env

def evaluate_model(model, env, n_episodes=10, deterministic=True, render=False):
    """
    Evaluate the model for a given number of episodes.
    Returns list of episode statistics.
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                try:
                    env.render()
                except:
                    pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Store episode data
        ep_dict = {
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length),
        }
        
        # Try to get additional info from the last step
        if info and len(info) > 0:
            for key in ['lines', 'hard_drops', 'soft_drops', 'rotations', 'left_moves', 'right_moves']:
                if key in info[0]:
                    ep_dict[key] = int(info[0][key])
        
        episode_data.append(ep_dict)
        
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    return episode_data, episode_rewards, episode_lengths

def main():
    args = parse_args()
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists() and not (model_path.with_suffix('.zip')).exists():
        print(f"[ERROR] Model not found at: {args.model}")
        print("Make sure you've trained a model first.")
        return
    
    print("="*60)
    print("EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Algorithm:     {args.algo.upper()}")
    print(f"Model:         {args.model}")
    print(f"Environment:   {args.env}")
    print(f"Persona:       {args.persona}")
    print(f"Episodes:      {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render:        {args.render}")
    print("="*60 + "\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    print("Creating evaluation environment...")
    env = make_eval_env(args.env, args.persona, args.seed, args.render)
    
    # Load model
    print(f"Loading model from {args.model}...")
    Algo = ALGOS[args.algo]
    try:
        model = Algo.load(args.model, env=env)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        env.close()
        return
    
    # Evaluate
    print(f"Running evaluation for {args.n_episodes} episodes...\n")
    episode_data, episode_rewards, episode_lengths = evaluate_model(
        model, env, 
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    max_length = np.max(episode_lengths)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes:      {args.n_episodes}")
    print(f"Mean Reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:    {min_reward:.2f}")
    print(f"Max Reward:    {max_reward:.2f}")
    print(f"Mean Length:   {mean_length:.1f} ± {std_length:.1f}")
    print(f"Max Length:    {max_length:.0f}")
    print("="*60 + "\n")
    
    # Save results if requested
    if args.save_results:
        results = {
            'model': args.model,
            'algorithm': args.algo,
            'persona': args.persona,
            'n_episodes': args.n_episodes,
            'deterministic': args.deterministic,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'min_reward': float(min_reward),
            'max_reward': float(max_reward),
            'mean_length': float(mean_length),
            'std_length': float(std_length),
            'max_length': int(max_length),
            'episodes': episode_data
        }
        
        # Ensure directory exists
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {args.save_results}")
    
    env.close()
    print("Evaluation complete!")

if __name__ == '__main__':
    main()