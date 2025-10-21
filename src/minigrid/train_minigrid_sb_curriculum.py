import argparse
import sys
import gym
import wandb
import numpy as np
import minigrid
from combined_env import CombinedMinigridEnv, MinigridFeaturesExtractor, CombinedMinigridEnv_Symbol
import yaml
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from gym import Wrapper
# from plot import plot_success_rates

# Optionally import WandbCallback from sb3_contrib if using an older version
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

SYMBOL = True

# Load the curriculum from a YAML file
def load_curriculum(config_path):
    with open(config_path, 'r') as f:
        curriculum = yaml.safe_load(f)
    return curriculum['curriculum']


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum = curriculum
        self.current_step = 0
        self.current_tasks = None  # Start with None

    def _on_step(self) -> bool:
        tasks_cache = self.current_tasks
        total_steps = self.num_timesteps
        for stage in self.curriculum:
            if total_steps >= stage['steps']:
                tasks_cache = stage['tasks']
            if total_steps < stage['steps']:
                break
        if tasks_cache != self.current_tasks:
            self.current_tasks = tasks_cache
            print(f"Changing tasks to: {self.current_tasks}")
            self._update_env_tasks()
        return True

    def _update_env_tasks(self):
        if isinstance(self.current_tasks, list):
            for env in self.model.get_env().envs:
                env.env._legal_tasks = self.current_tasks
            # print(self.model.get_env().envs[0].env.env.env._legal_tasks) # dummyvec/orderenforcing/craftenv
        else:
            for env in self.model.get_env().envs:
                env.env._legal_tasks = None

# Set train frequency to a certain number of steps
train_freq = TrainFreq(1000, unit=TrainFrequencyUnit.STEP)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Define the learning rate schedule
def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

def main(args):

     # Load curriculum from YAML file
    curriculum = load_curriculum(args.curriculum_file)

    # Define the make_env function for creating multiple environments
    def make_env(rank):
        def _init():
            if SYMBOL:
                env = CombinedMinigridEnv_Symbol()
            else:
                env = CombinedMinigridEnv()
            env = Monitor(env)
            return env
        return _init

    num_envs = args.n_envs  # Number of parallel environments

    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    # Initialize wandb for logging
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        sync_tensorboard=True,
        name=f"{args.wandb_run_name}-{args.algo}",
    )

    # Set up evaluation callback to periodically evaluate and save the model
    eval_callback = EvalCallback(
        env,  # Use the training environment for evaluation
        best_model_save_path=args.best_model_save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    curriculum_callback = CurriculumCallback(curriculum)
    callback = CallbackList([eval_callback, WandbCallback(verbose=1), curriculum_callback])

    # Verify the environment follows the Gym interface
    obs = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation shape:", obs[0].shape)  # Access the first environment's observation
    # check_env(env) # Commented out because check_env does not support VecEnvs

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
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    if args.net_arch:
        # Example for setting network architecture
        net_arch = [int(x) for x in args.net_arch.split(',')]
        policy_kwargs['net_arch'] = net_arch

    # Prepare model hyperparameters
    model_kwargs = {
        'env': env,
        'verbose': 1,
        # 'n_steps': MAX_STEPS,
        # 'batch_size': 2048,
        'exploration_initial_eps': 1.0,  # Start with 100% exploration, dqn
        'exploration_final_eps': 0.1,    # Reduce to 10% exploration, dqn
        'exploration_fraction': 0.9,     # dqn
        'tensorboard_log': args.tensorboard_log,
        # 'learning_rate': linear_schedule(args.learning_rate),
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
        # 'train_freq': 5,
        # 'train_freq': train_freq       # ddpg
        # 'n_steps': 10,
        # 'use_rms_prop': True
    }

    # Some algorithms have specific hyperparameters
    if args.algo.lower() in ['dqn', 'sac', 'td3']:
        model_kwargs['buffer_size'] = args.buffer_size

    # Instantiate the agent
    model = AlgoClass(
        args.policy,
        **model_kwargs,
        # policy_kwargs=policy_kwargs
    )

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Plot success rates after training
    # plot_success_rates(
    #     success_rates=task_success_callback.success_rates,
    #     eval_steps=task_success_callback.eval_steps,
    #     tasks=TASKS,
    #     save_path="success_rates.png"
    # )

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='ppo', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1e8, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=500000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='512, 256, 256, 128', help='Network architecture (comma-separated numbers)')
    parser.add_argument('--learning_tasks', type=str, default='All', help='Learning Tasks: All, Single, Multi')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=5e4, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='minigrid_sb_symbol_single', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')

    # Curriculum learning
    parser.add_argument('--curriculum_file', type=str, default='curriculum.yaml', help='Path to the curriculum YAML file')

    args = parser.parse_args()

    main(args)
