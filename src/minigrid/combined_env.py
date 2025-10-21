import gym.spaces
from minigrid.envs import *
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Key
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# sample from envs: EmptyEnv, FetchEnv, GoToDoorEnv, LockedRoomEnv, PlaygroundEnv

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class CombinedMinigridEnv(gym.Env):
    """
    Environment that random samples from a set of environments
    """
    def __init__(self, random_task=False, **kwargs):
        self.envs = [EmptyEnv(), CrossingEnv(size=8, max_steps=1000), FourRoomsEnv(), LockedRoomEnv(), PlaygroundEnv()]
        self.env = ImgObsWrapper(self.envs[0])
        super().__init__(**kwargs)
        self.action_space = gym.spaces.discrete.Discrete(8)
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype='uint8')
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.task_id = 0
        self._legal_tasks = [i for i in range(len(self.envs))]
        self.max_steps = self.env.max_steps
        self.step_count = 0
        self.random_task = random_task
        self.reset()
    
    def reset(self):
        self.step_count = 0
        if self.random_task:
            self.task_id = np.random.randint(len(self.envs))
            self.env = ImgObsWrapper(self.envs[self.task_id])
        else:
            self.task_id = np.random.choice(self._legal_tasks)
            self.env = ImgObsWrapper(self.envs[self.task_id])
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            # action = np.clip(action, self.action_space.low, self.action_space.high)
            action = np.argmax(action)
        obs, reward, done, truncated, info = self.env.step(action)
        self.step_count += 1
        done = (done or truncated)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        return self.env.render(mode, close)
    

class CombinedMinigridEnv_Symbol(gym.Env):
    """
    Environment that random samples from a set of environments
    """
    def __init__(self, random_task=False, **kwargs):
        self.envs = [
            EmptyEnv(), 
            CrossingEnv(size=9), 
            DoorKeyEnv(max_steps=1000), 
            FourRoomsEnv(max_steps=1000), 
            LockedRoomEnv(max_steps=1000), 
            PlaygroundEnv(max_steps=1000)
        ]
        self.env = FullyObsWrapper(self.envs[0])
        super().__init__(**kwargs)
        self.action_space = gym.spaces.discrete.Discrete(8)
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=(20, 20, 3), dtype='uint8')
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.task_id = 0
        self._legal_tasks = [i for i in range(len(self.envs))]
        self.max_steps = self.env.max_steps
        self.step_count = 0
        self.random_task = random_task
        self.reset()
    
    def reset(self):
        self.step_count = 0
        if self.random_task:
            self.task_id = np.random.randint(len(self.envs))
            self.env = FullyObsWrapper(self.envs[self.task_id])
        else:
            self.task_id = np.random.choice(self._legal_tasks)
            self.env = FullyObsWrapper(self.envs[self.task_id])
        obs, _ = self.env.reset()
        self.max_steps = self.env.max_steps
        obs = obs["image"]
        # pad to 20x20*3
        obs = np.pad(obs, ((0, 20-obs.shape[0]), (0, 20-obs.shape[1]), (0, 0)), mode='constant', constant_values=0)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            # action = np.clip(action, self.action_space.low, self.action_space.high)
            action = np.argmax(action)
        obs, reward, done, truncated, info = self.env.step(action)
        done = (done or truncated)
        
        obs = obs["image"]
        obs = np.pad(obs, ((0, 20-obs.shape[0]), (0, 20-obs.shape[1]), (0, 0)), mode='constant', constant_values=0)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        return self.env.render(mode, close)
    