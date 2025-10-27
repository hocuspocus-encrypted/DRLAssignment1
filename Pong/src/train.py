from __future__ import annotations
import argparse
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np
import json

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
    parser = argparse.ArgumentParser(description="Train DRL agent on Pong.")
    parser.add_argument('--algo', choices=list(ALGOS.keys()), default='ppo')
    parser.add_argument('--env', default='pong', help='environment name')
    parser.add_argument('--persona', choices=['survivor', 'explorer'], default='survivor', help='reward persona type')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--policy', default='CnnPolicy')
    parser.add_argument('--save', type=str, required=True, help='Path to save the model')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--use-subproc', action='store_true', 
                        help='Use SubprocVecEnv (faster but may have issues on macOS)')
    # PPO/A2C hyperparameters
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    return parser.parse_args()

def ensure_dir(path):
    """Ensure directory exists"""
    if path and path != '':
        os.makedirs(path, exist_ok=True)

class EpisodeLoggingCallback(BaseCallback):
    """
    Custom callback for logging episode statistics to CSV and JSON
    """
    def __init__(self, log_dir, algo, persona, verbose=0):
        super(EpisodeLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.algo = algo
        self.persona = persona
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_data = []
        
    def _on_step(self) -> bool:
        # Check if any episode finished in any of the environments
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                # Standard episode info
                episode_reward = ep_info.get('r', 0)
                episode_length = ep_info.get('l', 0)
                
                # Store episode data
                episode_dict = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'timestep': self.num_timesteps,
                }
                
                self.episode_data.append(episode_dict)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
        return True
    
    def _on_training_end(self) -> None:
        """Save data when training ends"""
        self.save_data()
    
    def save_data(self):
        """Save episode data to CSV and JSON"""
        if len(self.episode_data) == 0:
            print("No episode data to save")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.episode_data)
        
        # Save CSV
        csv_file = os.path.join(self.log_dir, f"{self.algo}_{self.persona}_episodes.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nSaved episode data to: {csv_file}")
        print(f"Total episodes logged: {len(df)}")
        
        # Save JSON
        json_file = os.path.join(self.log_dir, f"{self.algo}_{self.persona}_episodes.json")
        
        # Create summary statistics
        summary = {
            'algorithm': self.algo,
            'persona': self.persona,
            'total_episodes': len(df),
            'total_timesteps': int(self.num_timesteps),
            'mean_episode_reward': float(df['episode_reward'].mean()),
            'std_episode_reward': float(df['episode_reward'].std()),
            'min_episode_reward': float(df['episode_reward'].min()),
            'max_episode_reward': float(df['episode_reward'].max()),
            'mean_episode_length': float(df['episode_length'].mean()),
            'std_episode_length': float(df['episode_length'].std()),
            'episodes': self.episode_data
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved episode data to: {json_file}")
        print(f"\nSummary Statistics:")
        print(f"  Mean Reward: {summary['mean_episode_reward']:.2f} ± {summary['std_episode_reward']:.2f}")
        print(f"  Mean Length: {summary['mean_episode_length']:.2f} ± {summary['std_episode_length']:.2f}")

def make_vec_env(env_name, persona, n_envs, seed, use_subproc=False):
    """Create vectorized environment with persona support
    
    Args:
        env_name: Name of the environment
        persona: Reward persona type
        n_envs: Number of parallel environments
        seed: Random seed
        use_subproc: If True, use SubprocVecEnv (faster but can have issues on macOS)
                     If False, use DummyVecEnv (slower but more stable)
    """
    def make_env_fn(rank):
        def _init():
            # Check if build_env accepts persona parameter
            try:
                env = build_env(env_name, persona=persona)
            except TypeError:
                # Fallback if build_env doesn't accept persona
                print(f"Warning: build_env doesn't accept persona parameter")
                env = build_env(env_name)
            
            # Wrap with Monitor for tracking
            env = Monitor(env)
            # Set seed for this environment
            if hasattr(env, 'reset'):
                try:
                    env.reset(seed=seed + rank)
                except TypeError:
                    # Older gym API
                    env.seed(seed + rank)
                    env.reset()
            return env
        return _init

    set_random_seed(seed)
    
    # Use DummyVecEnv by default (more stable, especially on macOS)
    # SubprocVecEnv is faster but can have multiprocessing issues
    if use_subproc and n_envs > 1:
        print(f"Using SubprocVecEnv with {n_envs} processes")
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
    else:
        if n_envs > 1:
            print(f"Using DummyVecEnv with {n_envs} sequential environments (safer for macOS)")
        return DummyVecEnv([make_env_fn(i) for i in range(n_envs)])

def main():
    args = parse_args()
    
    # Create necessary directories
    print(f"Creating directories...")
    os.makedirs(args.logdir, exist_ok=True)
    
    # Ensure save directory exists
    save_dir = os.path.dirname(args.save)
    if save_dir:
        ensure_dir(save_dir)
    
    print(f"Setting random seed: {args.seed}")
    set_seed(args.seed)
    
    # Create environment with persona
    print(f"Creating {args.n_envs} parallel environments...")
    env = make_vec_env(args.env, args.persona, args.n_envs, args.seed, args.use_subproc)

    # For image-based environments like Pong, transpose images for CNN
    obs_shape = env.observation_space.shape
    print(f"Observation space shape: {obs_shape}")
    
    # Check if we need to transpose (channels last -> channels first)
    if len(obs_shape) == 3 and obs_shape[-1] in [1, 3, 4]:
        # Channels are last, need to transpose for PyTorch
        print("Applying VecTransposeImage to convert channels-last to channels-first...")
        env = VecTransposeImage(env)
        obs_shape = env.observation_space.shape
        print(f"New observation shape: {obs_shape}")

    # Select appropriate policy
    Algo = ALGOS[args.algo]
    if len(obs_shape) == 3:
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Algorithm:     {args.algo.upper()}")
    print(f"Policy:        {policy}")
    print(f"Environment:   {args.env}")
    print(f"Persona:       {args.persona}")
    print(f"Total steps:   {args.steps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Device:        {args.device}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gamma:         {args.gamma}")
    print(f"Save path:     {args.save}")
    print(f"{'='*60}\n")

    # Create model with hyperparameters
    model_kwargs = {
        'policy': policy,
        'env': env,
        'verbose': 1,
        'device': args.device,
        'tensorboard_log': args.logdir if args.tensorboard else None,
    }
    
    # Add algorithm-specific hyperparameters
    if args.algo in ['ppo', 'a2c']:
        model_kwargs.update({
            'n_steps': args.n_steps,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
        })
    
    print("Initializing model...")
    model = Algo(**model_kwargs)

    # Setup callbacks
    print("Setting up callbacks...")
    
    # Episode logging callback - logs detailed episode statistics
    episode_logger = EpisodeLoggingCallback(
        log_dir=args.logdir,
        algo=args.algo,
        persona=args.persona,
        verbose=1
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_freq = max(100_000 // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir if save_dir else '.',
        name_prefix='checkpoint',
        verbose=1,
    )

    # Evaluation callback - evaluates and saves best model
    print("Creating evaluation environment...")
    eval_env = make_vec_env(args.env, args.persona, 1, args.seed + 1000, use_subproc=False)
    
    # Apply same transformations to eval env
    if len(eval_env.observation_space.shape) == 3 and eval_env.observation_space.shape[-1] in [1, 3, 4]:
        eval_env = VecTransposeImage(eval_env)
    
    eval_freq = max(50_000 // args.n_envs, 1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir if save_dir else '.',
        log_path=args.logdir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Train
    print(f"\nStarting training...\n")
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[episode_logger, checkpoint_cb, eval_cb],
            progress_bar=True
        )
        
        print(f"\n{'='*60}")
        print(f"Training completed successfully!")
        print(f"{'='*60}")
        print(f"Saving final model to: {args.save}")
        model.save(args.save)
        
        # Ensure episode data is saved
        episode_logger.save_data()
        
        print(f"\nModel saved. You can load it with:")
        print(f"  model = {args.algo.upper()}.load('{args.save}')")
        
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user!")
        print("="*60)
        interrupted_path = args.save + "_interrupted"
        print(f"Saving current model to: {interrupted_path}")
        model.save(interrupted_path)
        # Save episode data even if interrupted
        episode_logger.save_data()
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR during training!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nClosing environments...")
        env.close()
        eval_env.close()
        print("Done!")

def save_training_data_to_csv(model, logdir, algo, persona):
    """
    Legacy function - episode data is now saved by EpisodeLoggingCallback
    This function is kept for backwards compatibility
    """
    print("\nNote: Episode data is automatically saved by EpisodeLoggingCallback")
    print(f"Check {logdir}/{algo}_{persona}_episodes.csv and .json files")

if __name__ == '__main__':
    main()