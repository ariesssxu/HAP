import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from simple_modules import ActorNet, ValueNet
import matplotlib.pyplot as plt
from collections import deque

import sys
# Import your custom environment
sys.path.append('..')
sys.path.append('../craft_env')
import craft_env
from craft_env.env_gym import TaskSamplingEnv, N_TASKS, TASKS, TASKS_SIMPLE_GET, TASKS_SIMPLE_MAKE, TASKS_HARD

from plot import plot_results

# Import wandb
import wandb

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Argument parsing
parser = argparse.ArgumentParser(description='Train an RL agent with multi-env support')
parser.add_argument('--n_envs', type=int, default=16, help='Number of parallel environments')
parser.add_argument('--visualize', action='store_true', help='Visualize the environment')
parser.add_argument('--random_reset_task', default=True, help='Randomly reset the task')
parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps per episode')
parser.add_argument('--n_steps', type=int, default=100, help='Number of steps per update per environment')
parser.add_argument('--n_episodes', type=int, default=1e4, help='Number of episodes to train')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--teacher_update_freq', type=int, default=500, help='Frequency of teacher updates')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--project_name', type=str, default='teacher_student_gan_rl', help='WandB project name')
parser.add_argument('--run_name', type=str, default='training_run', help='WandB run name')
args = parser.parse_args()

# Initialize wandb
wandb.init(project=args.project_name, name=args.run_name)

# Set constants
VISUALIZE = args.visualize
IF_RANDOM_RESET_TASK = args.random_reset_task
MAX_STEPS = args.max_steps
TASKS = TASKS_SIMPLE_GET + TASKS_SIMPLE_MAKE
N_TASKS = len(TASKS)
N_EPISODES = args.n_episodes
GAMMA = args.gamma
TEACHER_UPDATE_FREQ = args.teacher_update_freq
LR = args.lr
N_ENVS = args.n_envs
N_STEPS = args.n_steps

# Expected rewards per task
expected_rewards = [0.5 for _ in TASKS]  # Adjust as needed

# Function to create environments
def make_env(rank):
    def _init():
        env = gym.make(id='CraftEnv-v0', visualise=VISUALIZE, max_steps=MAX_STEPS)
        env = TaskSamplingEnv(env, TASKS, if_random_reset_task=IF_RANDOM_RESET_TASK)
        env = Monitor(env)
        return env
    return _init

# Generator (Teacher) Neural Network
class TaskGenerator(nn.Module):
    def __init__(self, n_tasks):
        super(TaskGenerator, self).__init__()
        # Learnable logits for task probabilities
        self.logits = nn.Parameter(torch.randn(n_tasks))
        
    def forward(self, n_samples=1):
        task_probs = torch.softmax(self.logits, dim=0)
        task_distribution = torch.distributions.Categorical(task_probs)
        task_idx = task_distribution.sample((n_samples,))
        log_prob = task_distribution.log_prob(task_idx)
        return task_idx, task_probs, log_prob

    def reset(self, probs=None):
        if probs is not None:
            # Convert provided probabilities to logits (inverse of softmax)
            with torch.no_grad():
                self.logits.copy_(torch.log(torch.tensor(probs)))
        else:
            # If no probabilities are provided, reset to default initialization
            with torch.no_grad():
                self.logits.copy_(torch.randn_like(self.logits))

def main():


    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # Get action and observation space from the environment
    n_actions = vec_env.action_space.n  # For discrete action spaces
    obs_shape = vec_env.observation_space.shape
    n_tasks = len(TASKS)
    gamma = args.gamma  # Discount factor

    # Initialize TaskGenerator and its optimizer
    task_generator = TaskGenerator(n_tasks)
    task_generator.reset(probs=[1.0 / n_tasks for _ in range(n_tasks)])  # Uniform distribution
    task_optimizer = optim.Adam(task_generator.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_net = ActorNet(obs_shape[0], hidden_dim=512, n_actions=n_actions).to(device)
    value_net = ValueNet(obs_shape[0], hidden_dim=512).to(device)

    optimizer = optim.Adam(list(actor_net.parameters()) + list(value_net.parameters()), lr=args.lr, betas=(0.9, 0.999))

    # Initialize environments and task indices
    obs = vec_env.reset()
    n_envs = N_ENVS
    task_indices = np.zeros(n_envs, dtype=int)

    # Sample initial tasks for all environments
    task_indices_tensor, _, _ = task_generator(n_envs)
    task_indices = task_indices_tensor.cpu().numpy()
    for idx, task_idx in enumerate(task_indices):
        vec_env.env_method('reset_task', TASKS[task_idx], indices=idx)

    obs = vec_env.reset()

    # Training Loop
    n_episodes = args.n_episodes
    gamma = args.gamma
    teacher_update_freq = args.teacher_update_freq

    episode_rewards = np.zeros(n_envs)
    task_success_rates = np.zeros(n_tasks)
    success_histories = [deque(maxlen=1000) for _ in range(n_tasks)]  # For smoothing

    # Initialize lists to store data for plotting
    episode_rewards_list = []
    success_rates_list = []
    task_indices_over_time = []

    eval_interval = 10000
    eval_success_rates_over_time = []
    task_success_rates_over_time = []
    eval_steps = []

    total_updates = n_episodes * MAX_STEPS // N_STEPS


    def create_eval_env(task_name, visualise=False, max_steps=1000):
        """
        Helper function to create an evaluation environment for a specific task.
        
        Args:
            task_name (str): The name of the task to set in the environment.
            visualise (bool): Whether to enable visualization.
            max_steps (int): Maximum steps per episode.
        
        Returns:
            gym.Env: The initialized Gym environment.
        """
        env = gym.make(id='CraftEnv-v0', visualise=visualise, max_steps=max_steps)
        env = TaskSamplingEnv(env, [task_name], if_random_reset_task=IF_RANDOM_RESET_TASK)
        env = Monitor(env)
        return env


    for update in trange(int(total_updates)):
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        for step in range(N_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action_probs = actor_net(obs_tensor)
            value = value_net(obs_tensor)

            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            actions.append(action)
            values.append(value.squeeze())
            log_probs.append(log_prob)

            action_cpu = action.cpu().numpy()
            obs, reward, done, info = vec_env.step(action_cpu)

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)

            # Update episode rewards and handle task resets
            for idx, d in enumerate(done):
                episode_rewards[idx] += reward[idx]
                if d:
                    # Append the episode reward to the list
                    episode_rewards_list.append(episode_rewards[idx])
            
                    # Compute success
                    expected_reward = expected_rewards[task_indices[idx]]
                    success = 1 if episode_rewards[idx] >= expected_reward else 0
                    # Update success history
                    success_histories[task_indices[idx]].append(success)

                    # Append success to success_rates_list
                    success_rates_list.append(success)
                    # Append task index to task_indices_over_time
                    task_indices_over_time.append(task_indices[idx])

                    # Reset episode reward
                    episode_rewards[idx] = 0.0
                    # Sample new task index
                    task_idx_tensor, _, _ = task_generator()
                    task_idx = task_idx_tensor.item()
                    task_indices[idx] = task_idx
                    # Reset the task in the environment
                    vec_env.env_method('reset_task', TASKS[task_idx], indices=idx)
                    # Reset the environment to get the initial observation
                    obs_reset = vec_env.env_method('reset', indices=idx)
                    obs[idx] = obs_reset[0]  # Update the observation for the reset environment

        # Convert collected data to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # Shape (n_steps, n_envs)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)  # Shape (n_steps, n_envs)
        values = torch.stack(values)  # Shape (n_steps, n_envs)
        log_probs = torch.stack(log_probs)  # Shape (n_steps, n_envs)

        # Compute returns and advantages
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        next_value = value_net(obs_tensor).detach().squeeze()
        returns = torch.zeros(N_STEPS + 1, n_envs).to(device)
        returns[-1] = next_value
        for step in reversed(range(N_STEPS)):
            returns[step] = rewards[step] + gamma * returns[step + 1] * (1 - dones[step])
        returns = returns[:-1]
        advantages = returns - values

        # Compute actor and critic losses
        actor_loss = - (log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # Optimize the actor and critic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update task success rates
        for task_idx in range(n_tasks):
            if len(success_histories[task_idx]) > 0:
                task_success_rates[task_idx] = np.mean(success_histories[task_idx])
            else:
                task_success_rates[task_idx] = 0.0

        # Compute the teacher's loss
        _, task_probs, _ = task_generator()

        # Convert success rates to tensor
        success_rates_tensor = torch.tensor(task_success_rates, dtype=torch.float32).to(device)

        # Compute the entropy of the task probabilities
        entropy = -torch.sum(task_probs * torch.log(task_probs + 1e-8))

        # Set the entropy regularization coefficient
        entropy_coeff = 0.1

        # Modify the teacher's loss to include entropy regularization
        teacher_loss = torch.dot(task_probs.to(device), success_rates_tensor) - entropy_coeff * entropy

        # Update the teacher every N steps
        if (update + 1) % teacher_update_freq == 0:
            task_optimizer.zero_grad()
            teacher_loss.backward()
            task_optimizer.step()

        # Log metrics to wandb
        wandb.log({
            'update': update + 1,
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
        })

        # Log success rates per task
        for task_idx in range(n_tasks):
            wandb.log({f'success_rate_task_{task_idx}': task_success_rates[task_idx]})

        # Evaluate every eval_interval updates
        if (update + 1) % (eval_interval // N_STEPS) == 0:
            # Log and store current task success rates
            task_success_rates_over_time.append(task_success_rates.copy())
            print(f"Update {update + 1}:")
            print(f"Task Success Rates: {task_success_rates}")
        
            # Initialize a list to store success rates for each task
            success_rates_per_task = []
        
            # Iterate over each task and perform parallel evaluations
            for task_id, task_name in enumerate(TASKS):
                # Create a list of environment constructors for parallel evaluation
                env_fns = [lambda task=task_name: create_eval_env(task) for _ in range(n_envs)]
                eval_vec_env = SubprocVecEnv(env_fns)
        
                # Reset the task for all environments
                eval_vec_env.env_method('reset_task', task_name, indices=range(n_envs))
                obs = eval_vec_env.reset()
                dones = np.zeros(n_envs, dtype=bool)  # Track completion of episodes
                episode_rewards_eval = np.zeros(n_envs)
        
                while not np.all(dones):
                    # Convert observations to tensors and move to the specified device
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        
                    # Get action probabilities from the actor network
                    with torch.no_grad():
                        action_probs = actor_net(obs_tensor)
        
                    # Select the action with the highest probability
                    actions = torch.argmax(action_probs, dim=1).cpu().numpy()
        
                    # Take a step in the environment
                    obs, rewards, done, info = eval_vec_env.step(actions)
        
                    # Update rewards and done flags
                    episode_rewards_eval += rewards * (~dones)
                    dones = np.logical_or(dones, done)
        
                    # Optional: Add a break condition to prevent infinite loops
                    if np.all(dones):
                        break
        
                # Determine the number of successful episodes
                successes = np.sum(episode_rewards_eval >= expected_rewards[task_id])
        
                # Calculate the success rate for the current task
                success_rate = successes / n_envs
                success_rates_per_task.append(success_rate)
        
                # Clean up the evaluation environment
                eval_vec_env.close()
        
                # Log the success rate to wandb (optional)
                wandb.log({f'eval_success_rate_{task_name}': success_rate, 'eval_step': update + 1})
        
            # Store the evaluation results
            eval_success_rates_over_time.append(success_rates_per_task)
            eval_steps.append(update + 1)
            print(f"Evaluation Success Rates (per task): {success_rates_per_task}")

            # Print Teacher's task probabilities
            with torch.no_grad():
                task_probs = torch.softmax(task_generator.logits, dim=0)
                print(f"Teacher's Task Probabilities: {task_probs.cpu().numpy()}")

            # Log evaluation metrics to wandb
            eval_metrics = {f'eval_success_rate_task_{i}': success_rates_per_task[i] for i in range(n_tasks)}
            wandb.log(eval_metrics)

    # Finish wandb run
    wandb.finish()

    # Plotting (Optional)
    plot_results(
        episode_rewards=episode_rewards_list,
        success_rates_list=success_rates_list,
        task_indices_over_time=task_indices_over_time,
        task_success_rates_over_time=task_success_rates_over_time,
        eval_success_rates_over_time=eval_success_rates_over_time,
        eval_steps=eval_steps,
        n_tasks=n_tasks,
        tasks=TASKS,
        save_path="teacher_student_gan_rl_with_teacher.png"
    )

if __name__ ==  '__main__':
    main()