from envs import PacmanEnv
import matplotlib.pyplot as plt
import numpy as np
import gym
from PIL import Image

env = PacmanEnv(render_mode='rgb_array', level = 4)
env.reset()
env.step(1)
# render and show
img = env.render()
img = Image.fromarray(img)
img.save("level_4.png")