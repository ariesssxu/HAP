import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torch
import gymnasium as gym
import numpy as np
from envs.pacman_env import PacmanEnv
from tqdm import trange
import time
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo
import tianshou as ts

np.set_printoptions(threshold=np.inf)

# env_train = PacmanEnv(render_mode="human")
env_test = gym.make("envs/PacMan-v0", render_mode="symbol")
env_test.reset()
log_file = "log.txt"
save_path = "video"
train_flag = "dqn-symbol-with-alive-panlty-1"
video_writer = imageio.get_writer(f"{save_path}/{train_flag}.mp4", fps=24)
test_steps = 1000

space_info = SpaceInfo.from_env(env_test)
state_shape = space_info.observation_info.obs_shape
action_shape = space_info.action_info.action_shape
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128, 128])
policy: ts.policy.DQNPolicy = ts.policy.DQNPolicy(
        model=net,
        optim=None,
        action_space=env_test.action_space,
    )
policy.load_state_dict(torch.load(f"log/{train_flag}/checkpoint.pth")["model"], strict=False)

for i in range(10):
    state = env_test.reset()[0]
    done = False
    for i in trange(test_steps):
        action = policy.model(state.reshape(1, -1))[0].argmax(dim=1).item()
        state, reward, done, _, _ = env_test.step(action)
        print(action, reward, done)
        env_test.game.render(mode="rgb_array")
        rgb_obs = env_test.game.get_observation(mode="rgb_array")
        video_writer.append_data(rgb_obs)
        if done:
            break
    
    for t in range(10):
        empty_obs = np.zeros_like(rgb_obs)
        video_writer.append_data(empty_obs)


env_test.close()