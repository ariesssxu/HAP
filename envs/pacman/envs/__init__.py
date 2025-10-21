import gymnasium as gym
from .pacman_env import PacmanEnv
from gymnasium.envs.registration import register

register(
    id='envs/PacMan-v0',
    entry_point='envs.pacman_env:PacmanEnv',
    max_episode_steps=1000,
)