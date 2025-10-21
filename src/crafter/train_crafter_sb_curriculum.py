import argparse
import gym
import crafter
import yaml
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
import wandb
import sys
import numpy as np
import os

# Optionally import WandbCallback from sb3_contrib if using an older version
from wandb.integration.sb3 import WandbCallback

MAX_STEPS = 10000
TASKS = None

# Load the curriculum from a YAML file
def load_curriculum(config_path):
    with open(config_path, 'r') as f:
        curriculum = yaml.safe_load(f)
    return curriculum['curriculum']

# Custom Callback to adjust tasks based on curriculum
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum = curriculum
        self.current_step = 0
        self.current_tasks = self.curriculum[0]['tasks']  # Start with the first tasks

    def _on_step(self) -> bool:
        # Update the tasks based on the number of steps trained
        total_steps = self.num_timesteps
        
        # Progress through curriculum based on total steps
        for stage in self.curriculum:
            if total_steps >= stage['steps'] and stage['tasks'] != self.current_tasks:
                self.current_tasks = stage['tasks']
                print(f"Changing tasks to: {self.current_tasks}")
                self._update_env_tasks()
                break
        return True

    def _update_env_tasks(self):
        # This method will update the environment with the new tasks
        self.model.get_env()._legal_tasks = self.current_tasks
        print(self.model.get_env()._legal_tasks)

def main(args):
    # Load curriculum from YAML file
    curriculum = load_curriculum(args.curriculum_file)

    # Define the make_env function for creating multiple environments
    def make_env(rank):
        def _init():
            # Create the Crafter environment with dynamic tasks
            env = gym.make('CrafterReward-v1', tasks=TASKS)  # Default tasks are set here
            env = crafter.Recorder(
                env, f'logs/crafter',
                save_stats=False,
                save_video=False,
                save_episode=False,
            )
            env = Monitor(env)
            return env
        return _init

    num_envs = args.n_envs  # Number of parallel environments

    # Create the vectorized environment
    envs = gym.make('CrafterReward-v1', tasks=TASKS)
    envs = Monitor(envs)

    # Initialize wandb for logging
    wandb.init(
        project=f"{args.wandb_project}-{args.algo}",
        config=vars(args),
        sync_tensorboard=True,
        name=args.wandb_run_name,
    )

    # Set up evaluation callback to periodically evaluate and save the model
    eval_callback = EvalCallback(
        envs,  # Use the training environment for evaluation
        best_model_save_path=args.best_model_save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    # Create a custom curriculum callback
    curriculum_callback = CurriculumCallback(curriculum)
    callback = CallbackList([eval_callback, WandbCallback(verbose=1), curriculum_callback])

    # Dictionary to map algorithm names to constructors
    ALGOS = {
        'dqn': DQN,
        'ppo': PPO,
        'a2c': A2C,
        'sac': SAC,
        'td3': TD3,
    }

    # Ensure the selected algorithm is valid
    assert args.algo.lower() in ALGOS.keys(), f"Invalid algorithm. Choose from {list(ALGOS.keys())}"

    # Select the algorithm class
    AlgoClass = ALGOS[args.algo.lower()]

    # Create policy keyword arguments
    policy_kwargs = {}
    if args.net_arch:
        net_arch = [int(x) for x in args.net_arch.split(',')]
        policy_kwargs['net_arch'] = net_arch

    # Prepare model hyperparameters
    model_kwargs = {
        'env': envs,
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
    }

    if args.algo.lower() in ['dqn', 'sac', 'td3']:
        model_kwargs['buffer_size'] = args.buffer_size

    # Instantiate the agent
    model = AlgoClass(
        args.policy,
        **model_kwargs,
        policy_kwargs=policy_kwargs
    )

    # Setup logger
    log_path = "./logs/sb3_log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=1
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='ppo', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1e6, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='512, 512, 256, 256, 128', help='Network architecture (comma-separated numbers)')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=5e4, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='acrl-test-sb-crafter-curriculum', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')

    # Curriculum learning
    parser.add_argument('--curriculum_file', type=str, default='curriculum.yaml', help='Path to the curriculum YAML file')

    args = parser.parse_args()

    main(args)