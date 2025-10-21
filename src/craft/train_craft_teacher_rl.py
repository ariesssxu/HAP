import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from teacher_env_rl import TeacherEnv
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
import gym
import sys
# Import your custom environment
sys.path.append('..')
sys.path.append('../craft_env')
import craft_env
from craft_env.env_gym import TaskSamplingEnv, N_TASKS, TASKS, TASKS_SIMPLE_GET, TASKS_SIMPLE_MAKE, TASKS_HARD
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from plot import plot_results

# Import wandb
import wandb

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Argument parsing
parser = argparse.ArgumentParser(description='Train an RL agent with a meta-learning teacher')
parser.add_argument('--n_envs', type=int, default=16, help='Number of parallel environments (set to 1 for episodic training)')
parser.add_argument('--visualize', action='store_true', help='Visualize the environment')
parser.add_argument('--random_reset_task', default=True, help='Randomly reset the task')
parser.add_argument('--n_steps', type=int, default=100, help='Maximum number of steps per episode')
parser.add_argument('--env_steps', type=int, default=100000, help='Number of episodes to train')
parser.add_argument('--total_timesteps', type=int, default=1e8, help='Total number of timesteps to train the teacher')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--teacher_lr', type=float, default=1e-4, help='Learning rate for the teacher')
parser.add_argument('--agent_lr', type=float, default=1e-3, help='Learning rate for the agent')
parser.add_argument('--project_name', type=str, default='meta_teacher_rl', help='WandB project name')
parser.add_argument('--run_name', type=str, default='training_run', help='WandB run name')
args = parser.parse_args()

# Initialize wandb
# wandb.init(project=args.project_name, name=args.run_name)
sample_env = TaskSamplingEnv(gym.make('CraftEnv-v0'), TASKS_SIMPLE_GET, if_random_reset_task=True)
obs_dim = sample_env.observation_space.shape[0]
n_actions = sample_env.action_space.n

# Set constants
TASKS = TASKS_SIMPLE_GET + TASKS_SIMPLE_MAKE
n_tasks = len(TASKS)
TEACHER_LR = args.teacher_lr
AGENT_LR = args.agent_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expected rewards per task
expected_rewards = [0.5 for _ in TASKS]  # Adjust as needed


# Define agent configuration
agent_config = {
    'obs_dim': obs_dim,               # Number of performance metrics
    'n_actions': n_actions,             # Number of tasks
    'lr': AGENT_LR,                   # Learning rate for the agent           # Number of tasks to train on per teacher step
    'performance_window': 100,        # Window size for moving average
    'visualize': args.visualize,           # Visualization flag
    'if_random_reset_task': args.random_reset_task,  # Task reset flag
    'agent_gamma': args.gamma,              # Discount factor for the agent
    'n_envs': args.n_envs,                 # Number of parallel environments
    'n_steps': args.n_steps,                        # Number of steps per environment
}

def make_env():
    def _init():
        env = TeacherEnv(agent_config=agent_config, task_list=TASKS, expected_rewards=expected_rewards, env_steps=args.env_steps, device=device)
        return env
    return _init


def main():
    # Initialize TeacherEnv
    # teacher_env = TeacherEnv(agent_config=agent_config, task_list=TASKS, expected_rewards=expected_rewards, env_steps=args.env_steps, device=device)
    teacher_env = SubprocVecEnv([make_env() for i in range(4)])
    # Optional: Check the environment for compatibility
    # check_env(teacher_env, warn=True)

    # Initialize the PPO model for the teacher
    teacher_model = PPO(
        policy='MlpPolicy',
        env=teacher_env,
        learning_rate=3e-4,
        n_steps=args.env_steps,                # Each step is one teacher episode
        batch_size=32,
        verbose=1,
        tensorboard_log="./teacher_tensorboard/"
    )

    # Train the teacher agent
    teacher_model.learn(
        total_timesteps=args.total_timesteps
    )  # Adjust the number of timesteps as needed

    # Save the trained teacher model
    teacher_model.save("ppo_teacher_model")

    # To load the model later
    # teacher_model = PPO.load("ppo_teacher_model")

    # Finish wandb run
    # wandb.finish()

# # Plotting (Optional)
# plot_results(
#     episode_rewards=teacher_env.agent['episode_rewards_list'],
#     success_rates_list=teacher_env.agent['success_rates_list'],
#     task_indices_over_time=teacher_env.agent['task_indices_over_time'],
#     task_success_rates_over_time=teacher_env.performance_metrics_over_time,
#     eval_success_rates_over_time=teacher_env.eval_success_rates_over_time,
#     eval_steps=teacher_env.eval_steps,
#     n_tasks=n_tasks,
#     tasks=TASKS,
#     save_path="meta_teacher_rl_results.png"
# )

if __name__ ==  '__main__':
    main()