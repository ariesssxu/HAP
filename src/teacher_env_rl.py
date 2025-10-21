import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from simple_modules import ActorNet, ValueNet
from stable_baselines3.common.monitor import Monitor
import sys
from collections import deque
from tqdm import trange
# Import your custom environment
sys.path.append('..')
sys.path.append('../craft_env')
from craft_env.env_gym import TaskSamplingEnv
from stable_baselines3.common.vec_env import DummyVecEnv


class TeacherEnv(gym.Env):
    """
    Gym environment for the RL meta-learning teacher.
    The teacher assigns a probability distribution over tasks,
    the agent trains on tasks sampled from this distribution,
    and receives a reward based on the agent's performance.
    """
    def __init__(self, agent_config, task_list, expected_rewards, env_steps,  device):
        super(TeacherEnv, self).__init__()
        
        self.task_list = task_list
        self.n_tasks = len(task_list)
        self.expected_rewards = expected_rewards
        self.device = device
        self.episode_steps = 0
        self.env_steps = env_steps
        
        # Define action space: Probability distribution over tasks
        # Using Box space with values between 0 and 1, shape (n_tasks,)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_tasks,), dtype=np.float32)
        
        # Define observation space: Agent's performance metrics (success rates per task)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_tasks,), dtype=np.float32)
        
        # Agent configuration (to be passed to the agent training loop)
        self.agent_config = agent_config
        
        # Initialize performance metrics
        self.performance_metrics = np.zeros(self.n_tasks, dtype=np.float32)
        
        # Initialize the agent (student) components
        self.agent = self.initialize_agent()
    
    def initialize_agent(self):
        """
        Initialize the agent's networks and optimizer.
        This function should mirror the agent's initialization in the original code.
        """
        # Assuming ActorNet and ValueNet are defined elsewhere and imported
        actor_net = ActorNet(self.agent_config['obs_dim'], hidden_dim=512, n_actions=self.agent_config['n_actions']).to(self.device)
        value_net = ValueNet(self.agent_config['obs_dim'], hidden_dim=512).to(self.device)
        optimizer = optim.Adam(list(actor_net.parameters()) + list(value_net.parameters()), lr=self.agent_config['lr'], betas=(0.9, 0.999))

        # Create environments for the sampled tasks
        self.env_fns = [self.make_env(index=i) for i in range(self.agent_config['n_envs'])]
        self.vec_env = DummyVecEnv(self.env_fns)
        
        
        return {
            'actor_net': actor_net,
            'value_net': value_net,
            'optimizer': optimizer
        }
    
    def reset(self):
        """
        Reset the environment for a new teacher episode.
        Returns the initial observation (performance metrics).
        """
        # Reset performance metrics
        self.performance_metrics = np.zeros(self.n_tasks, dtype=np.float32)
        
        # Reset the agent's parameters
        self.reset_agent()

        # reset the environments
        self.reset_envs()
        
        return self.performance_metrics.copy()
    
    def step(self, action):
        """
        Execute one step in the environment:
        - Assign task probabilities based on the teacher's action.
        - Train the agent on tasks sampled from these probabilities.
        - Evaluate the agent's performance.
        - Compute the reward.
        
        Args:
            action (np.ndarray): Probability distribution over tasks.
        
        Returns:
            observation (np.ndarray): Updated performance metrics.
            reward (float): Reward signal.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        # Ensure action is a valid probability distribution
        task_probs = np.clip(action, 1e-6, 1.0)  # Avoid zero probabilities
        task_probs /= task_probs.sum()
        
        # Sample tasks for the agent based on task_probs
        sampled_tasks = np.random.choice(self.task_list, size=self.agent_config['n_envs'], p=task_probs)
        print(len(sampled_tasks))
        
        # Train the agent on the sampled tasks from scratch
        self.train_agent(sampled_tasks)
        
        # Evaluate the agent's performance
        self.evaluate_agent()
        
        # Compute reward based on performance
        reward = self.compute_reward()

        print(f"Reward: {reward}, Performance: {self.performance_metrics}")
        
        # The observation is the updated performance metrics
        observation = self.performance_metrics.copy()

        # if reward >= 0.5:
        #     reward += 100
        
        # Episode is done after one step
        done = True if self.episode_steps >= self.env_steps else False
        self.episode_steps += 1
        
        info = {}
        
        return observation, reward, done, info
    
    def reset_agent(self):
        """
        Reset the agent's networks to initial random weights.
        """
        def reset_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.agent['actor_net'].apply(reset_weights)
        self.agent['value_net'].apply(reset_weights)
        # Optionally, reset optimizer state if necessary

    def reset_envs(self):
        """
        Reset the environments for the agent training loop.
        """
        obs = self.vec_env.reset()
        return obs
    
    def train_agent(self, tasks):
        """
        Train the agent on the given list of tasks.
        This function should encapsulate the agent's training loop.
        
        Args:
            tasks (list): List of tasks to train on.
        """
        # Implement the agent's training loop here
        # This should mirror the original agent training loop but focus on the given tasks
        # For example:

        # Initialize environments
        for idx, task in enumerate(tasks):
            self.vec_env.env_method('reset_task', task, indices=idx)
        obs = self.vec_env.reset()
        
        gamma = self.agent_config['agent_gamma']

        episode_rewards_list = []
        success_rates_list = []

        episode_rewards = np.zeros(self.agent_config['n_envs'])

        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        for step in range(self.agent_config['n_steps']):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action_probs = self.agent['actor_net'](obs_tensor)
            value = self.agent['value_net'](obs_tensor)

            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            actions.append(action)
            values.append(value.squeeze())
            log_probs.append(log_prob)

            action_cpu = action.cpu().numpy()
            obs, reward, done, info = self.vec_env.step(action_cpu)

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
                    success = 1 if episode_rewards[idx] >= 0.5 else 0
                    # Update success history
                    # success_histories[task_indices[idx]].append(success)

                    # Append success to success_rates_list
                    success_rates_list.append(success)
                    # Append task index to task_indices_over_time
                    # task_indices_over_time.append(task_indices[idx])

                    # Reset episode reward
                    episode_rewards[idx] = 0.0

                    # obs_reset = self.vec_env.env_method('reset', indices=idx)
                    # obs[idx] = obs_reset[0]

        # Convert collected data to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # Shape (n_steps, n_envs)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)  # Shape (n_steps, n_envs)
        values = torch.stack(values)  # Shape (n_steps, n_envs)
        log_probs = torch.stack(log_probs)  # Shape (n_steps, n_envs)

        # Compute returns and advantages
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_value = self.agent['value_net'](obs_tensor).detach().squeeze()
        returns = torch.zeros(self.agent_config['n_steps'] + 1, self.agent_config['n_envs']).to(self.device)
        returns[-1] = next_value
        for step in reversed(range(self.agent_config['n_steps'])):
            returns[step] = rewards[step] + gamma * returns[step + 1] * (1 - dones[step])
        returns = returns[:-1]
        advantages = returns - values

        # Compute actor and critic losses
        actor_loss = - (log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # Optimize the actor and critic
        self.agent['optimizer'].zero_grad()
        loss.backward()
        self.agent['optimizer'].step()
            
    def make_env(self, task=None, index=None):
        """
        Create a single environment instance for a specific task.
        
        Args:
            task (str): The task to assign to the environment.
        
        Returns:
            Callable: A function that initializes the environment.
        """
        def _init():
            env = gym.make(id='CraftEnv-v0', visualise=self.agent_config['visualize'], max_steps=self.agent_config['n_steps'])
            if task is not None:
                env = TaskSamplingEnv(env, [task], if_random_reset_task=self.agent_config['if_random_reset_task'])
            else:
                env = TaskSamplingEnv(env, self.task_list, if_random_reset_task=self.agent_config['if_random_reset_task'])
            env = Monitor(env)
            return env
        return _init
    
    def evaluate_agent(self):
        """
        Evaluate the agent's performance on the given tasks.
        Updates the performance_metrics based on evaluation.
        
        Args:
            tasks (list): List of tasks to evaluate on.
        """
        # Implement evaluation logic
        # For example, measure success rates on each task
        
        # Initialize a list to store success rates for each task
        success_rates_per_task = []
    
        # Iterate over each task and perform parallel evaluations
        for task_id, task_name in enumerate(self.task_list):
            # Create a list of environment constructors for parallel evaluation
            env_fns = [self.make_env(task_name) for _ in range(self.agent_config['n_envs'])]
            eval_vec_env = DummyVecEnv(env_fns)
    
            # Reset the task for all environments
            eval_vec_env.env_method('reset_task', task_name, indices=range(self.agent_config['n_envs']))
            obs = eval_vec_env.reset()
            dones = np.zeros(self.agent_config['n_envs'], dtype=bool)  # Track completion of episodes
            episode_rewards_eval = np.zeros(self.agent_config['n_envs'])
    
            while not np.all(dones):
                # Convert observations to tensors and move to the specified device
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
    
                # Get action probabilities from the actor network
                with torch.no_grad():
                    action_probs = self.agent['actor_net'](obs_tensor)
    
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
            successes = np.sum(episode_rewards_eval >= self.expected_rewards[task_id])
    
            # Calculate the success rate for the current task
            success_rate = successes / self.agent_config['n_envs']
            success_rates_per_task.append(success_rate)
    
            # Clean up the evaluation environment
            eval_vec_env.close()
        
        # Update performance metrics as moving averages
        for idx in range(self.n_tasks):
            # Simple moving average with a fixed window (e.g., over episodes)
            self.performance_metrics[idx] = 0.9 * self.performance_metrics[idx] + 0.1 * success_rates_per_task[idx]
    
    def compute_reward(self):
        """
        Compute the reward for the teacher based on the agent's performance.
        This can be designed to encourage high performance across tasks.
        
        Returns:
            float: The computed reward.
        """
        # Example: Average success rate across all tasks
        reward = np.mean(self.performance_metrics)
        return reward