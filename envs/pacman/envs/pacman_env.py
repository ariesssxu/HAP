import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
from pacman_src.run import GameController

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["symbol", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, level=0):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode
        self.render_mode = "symbol" if render_mode is None else render_mode
        # Observations are indexes of the objects on the map.
        if render_mode == "rgb_array":
            self.observation_space = gym.spaces.Box(0, 255, (448, 576, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(0, 20, (36, 28, 2), dtype=np.int64)        
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.discrete.Discrete(5)
        self.level = level


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.game = GameController(level=self.level)
        self.game.startGame()
        self.max_step = 1000
        self.current_step = 0

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        self.game.restartGame(level=self.level)
        self.current_step = 0
        self.game.render(mode=self.render_mode)
        observation = self.game.get_observation(mode=self.render_mode)
        info = {}
        # plt.imsave("test.png", observation)
        return observation

    def step(self, action):
        self.game.step(action)
        self.game.render(mode=self.render_mode)
        observation = self.game.get_observation(mode=self.render_mode)
        reward = self.game.get_reward()
        terminated = self.game.is_terminated()
        done = terminated or self.current_step >= self.max_step
        info = {}
        self.current_step += 1
        # plt.imsave("test.png", observation)
        return observation, reward, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.game.render(self.render_mode)
        return self.game.get_observation(self.render_mode)

    def close(self):
        # if self.window is not None:
        pygame.display.quit()
        pygame.quit()

    def seed(self, seed):
        np.random.seed(seed)
