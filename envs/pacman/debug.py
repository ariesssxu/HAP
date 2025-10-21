import gymnasium as gym
import numpy as np
from envs.pacman_env import PacmanEnv
from tqdm import trange
import time

np.set_printoptions(threshold=np.inf)

# env_train = PacmanEnv(render_mode="human")
env_train = gym.make("envs/PacMan-v0", render_mode="symbol")
env_train.reset()
env_train.step(0)
num_episodes = 20
log_file = "log.txt"


# learning
for i in trange(num_episodes):
    state = env_train.reset()[0]
    done = False

    while not done:
        action = env_train.action_space.sample()
        print(action)
        next_state, reward, done, _, _ = env_train.step(action)
        # print(next_state)
        print(env_train.game.pause.paused)
        state = next_state
        time.sleep(1)

env_train.close()