import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt
from collections import deque

# Define action mapping
actions = ['left', 'right']
action_to_idx = {'left': 0, 'right': 1}
idx_to_action = {0: 'left', 1: 'right'}
n_actions = len(actions)

# Define tasks as sequences of actions (target sequences)
tasks = [
    ['left', 'right'],  # Task 0
    ['left', 'left', 'right', 'left'],  # Task 1
    ['right', 'right', 'left', 'left', 'left', 'right', 'right', 'right'],  # Task 2
    ['right', 'left', 'left', 'left', 'left', 'right', 'right', 'right'] * 2  # Task 3 (length 16)
]

n_tasks = len(tasks)

# Map tasks to indices for easier processing
task_sequences = [np.array([action_to_idx[a] for a in task]) for task in tasks]

# Expected rewards (set to 1.0 for simplicity)
expected_rewards = [1.0 for _ in tasks]

# Custom Gym Environment
class TaskEnv(gym.Env):
    def __init__(self, task_sequences, expected_rewards, max_steps=16):
        super(TaskEnv, self).__init__()
        self.task_sequences = task_sequences
        self.expected_rewards = expected_rewards
        self.n_tasks = len(task_sequences)
        # self.action_space = spaces.Discrete(n_actions)
        self.action_space = spaces.Box(0, 1, (n_actions,), np.float32)
        self.max_steps = max_steps
        self.current_step = 0
        self.solved = False
        self.actions_taken = []
        self.task_idx = None
        self.target_sequence = None
        self.task_flag = None  # One-hot encoding of the task index

        # Observation space: current step as one-hot encoding and task_flag
        self.observation_space = spaces.Dict({
            'current_step': spaces.Box(low=0.0, high=1.0, shape=(self.max_steps,), dtype=np.float32),
            'task_flag': spaces.Box(low=0.0, high=1.0, shape=(self.n_tasks,), dtype=np.float32),
            # Optionally include previous_action
        })

    def reset_task(self, task_idx):
        self.task_idx = task_idx
        self.target_sequence = self.task_sequences[task_idx]
        self.expected_reward = self.expected_rewards[task_idx]
        # Create one-hot encoding for the task flag
        self.task_flag = np.zeros(self.n_tasks, dtype=np.float32)
        self.task_flag[task_idx] = 1.0

    def reset(self):
        # raise Warning("Reset_task at each reset().")
        self.reset_task(np.random.randint(self.n_tasks))  # Randomly select a task
        if self.task_idx is None:
            self.task_idx = np.random.randint(self.n_tasks)
            self.reset_task(self.task_idx)
            # raise ValueError("Task not set. Please call reset_task(task_idx) before reset().")
        self.current_step = 0
        self.done = False
        self.solved = False
        self.actions_taken = []
        # Initialize observation
        observation = {
            'current_step': self._get_current_step_observation(),
            'task_flag': self.task_flag.copy(),
            # 'previous_action': np.zeros(n_actions, dtype=np.float32),
        }
        return observation

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            action = np.argmax(action)
        self.actions_taken.append(action)
        reward = 0.0
        self.current_step += 1

        # Check if action is correct
        correct_action = self.target_sequence[self.current_step - 1]
        if action == correct_action:
            if self.current_step == len(self.target_sequence):
                reward = self.expected_reward
                self.solved = True
                self.done = True
            else:
                reward = 0.1  # Optional intermediate reward
        else:
            self.done = True   # End episode on incorrect action
            self.solved = False

        if self.current_step >= self.max_steps:
            self.done = True

        # Prepare the next observation
        observation = {
            'current_step': self._get_current_step_observation(),
            'task_flag': self.task_flag.copy(),
            # 'previous_action': np.eye(n_actions)[action],
        }

        return observation, reward, self.done, {}

    def _get_current_step_observation(self):
        current_step_input = np.zeros(self.max_steps, dtype=np.float32)
        if self.current_step < self.max_steps:
            current_step_input[self.current_step] = 1.0
        return current_step_input

    def render(self, mode='human'):
        pass  # Implement visualization if needed
