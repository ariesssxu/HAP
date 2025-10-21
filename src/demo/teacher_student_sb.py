import argparse
import sys
import gym
import wandb
import numpy as np

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
from gym import Wrapper
# from plot import plot_success_rates

# Optionally import WandbCallback from sb3_contrib if using an older version
from wandb.integration.sb3 import WandbCallback
import os
import pandas as pd
from simple_env import TaskEnv, task_sequences, expected_rewards


class TaskRewardLogger(BaseCallback):
    def __init__(self, log_interval=1, log_file="logs/task_reward_log_sac.csv"):
        super(TaskRewardLogger, self).__init__()
        self.log_interval = log_interval
        self.log_file = log_file
        self.episode_rewards = []  # To store the rewards for each episode
        self.episode_task_ids = []  # To store the task ids for each episode

        # Check if the file exists, if not, create it with headers
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        df = pd.DataFrame(columns=["episode", "task_id", "reward"])
        df.to_csv(self.log_file, index=False)            

    def _on_step(self) -> bool:
        # Only log at the end of each episode
        env = self.training_env.envs[0].env
        done = self.locals.get("dones")[0]  # Assumes only one environment
        if done:
            # get env
            reward = self.locals.get("rewards")[0]
            task_id = env.task_idx  # Assumes task_id is in 'infos'
            if_solved = reward==1.0
            
            # Log to Wandb
            wandb.log({"task_id": task_id, "if_solved": if_solved}, step=self.n_calls)
            
            # Append to local file
            self.episode_rewards.append(if_solved)
            self.episode_task_ids.append(task_id)
            
            # Save to local file after each episode
            df = pd.DataFrame({
                "step": [self.num_timesteps],
                "task_id": [task_id],
                "solved": [if_solved]
            })
            df.to_csv(self.log_file, mode='a', header=False, index=False)

        return True

    def _on_training_end(self):
        # Optional: Do something after training ends (like printing a summary)
        print(f"Training completed. Task-reward log saved to {self.log_file}.")
        return super()._on_training_end()
    

# Define the learning rate schedule
def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

def main(args):
    
    env = TaskEnv(task_sequences, expected_rewards)
    env = Monitor(env)

    # Initialize wandb for logging
    wandb.init(
        project=f"{args.wandb_project}",
        name=f"{args.wandb_run_name}-{args.algo}",
        config=vars(args),
        sync_tensorboard=True,
    )

    # Combine callbacks: evaluation, wandb logging, and task success[] rates
    task_reward_callback = TaskRewardLogger(log_interval=1)  # Log every episode
    callback = CallbackList([WandbCallback(), task_reward_callback])
    
    # Verify the environment follows the Gym interface
    obs = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    # print("Initial observation shape:", obs[0].shape)  # Access the first environment's observation
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
    policy_kwargs = {}
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
        # 'exploration_initial_eps': 1.0,  # Start with 100% exploration, dqn
        # 'exploration_final_eps': 0.1,    # Reduce to 10% exploration, dqn
        # 'exploration_fraction': 0.1,     # dqn
        'tensorboard_log': args.tensorboard_log,
        # 'learning_rate': linear_schedule(args.learning_rate),
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
        # 'train_freq': 5, # dqn
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
        policy_kwargs=policy_kwargs
    )

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )

    # Save the trained agent
    model.save(args.model_save_path)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stable Baselines3 and wandb")

    # Algorithm and Environment arguments
    parser.add_argument('--algo', type=str, default='ppo', help='RL Algorithm: dqn, ppo, a2c, sac, td3')
    parser.add_argument('--policy', type=str, default='MultiInputPolicy', help='Policy type: MlpPolicy, CnnPolicy, etc.')

    # Hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=3e5, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=5000, help='Replay buffer size (for off-policy algorithms)')
    parser.add_argument('--net_arch', type=str, default='128', help='Network architecture (comma-separated numbers)')
    parser.add_argument('--learning_tasks', type=str, default='All', help='Learning Tasks: All, Single, Multi')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1e3, help='Evaluation frequency in timesteps')
    parser.add_argument('--best_model_save_path', type=str, default='./logs/best_model', help='Path to save the best model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Directory to save evaluation logs')

    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='simple_test', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='start')

    # Logging and saving paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/', help='TensorBoard log directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    args = parser.parse_args()

    main(args)
