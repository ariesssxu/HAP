import pygame
import gym
import numpy as np
import os
from minigrid.envs import *
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper
import time

# Initialize pygame
pygame.init()

tasks = [EmptyEnv(), CrossingEnv(), FourRoomsEnv(), LockedRoomEnv(), PlaygroundEnv()]

# Set up the window size (you can adjust this based on your needs)
WINDOW_WIDTH = 256
WINDOW_HEIGHT = 256
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Human-Interactive Gym Environment")

# Set up the clock to control the frame rate
clock = pygame.time.Clock()

# Create the environment (replace with your own environment paths)
env = RGBImgObsWrapper(tasks[1])
# Reset the environment
obs = env.reset()

done = False

total_reward = 0

preset_action_sequence = []

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Handling key events for movement
    keys = pygame.key.get_pressed()
    action = None  # Define action here based on key press
    
    if keys[pygame.K_a]:  # Move forward (you can adjust this)
        action = 0  # Replace with actual action index for 'w'
    elif keys[pygame.K_d]:  # Move left
        action = 1  # Replace with actual action index for 'a'
    elif keys[pygame.K_w]:  # Move backward
        action = 2  # Replace with actual action index for 's'
    elif keys[pygame.K_s]:  # Move backward
        action = 7  # Replace with actual action index for 's'
    elif keys[pygame.K_o]:  # Move right
        action = 3  # Replace with actual action index for 'd'
    elif keys[pygame.K_p]:  # Move right
        action = 4  # Replace with actual action index for 'd'
    elif keys[pygame.K_l]:  # Move right
        action = 5 
    elif keys[pygame.K_ESCAPE]:  # Interact (e.g., select)
        action = 6  # Replace with actual action index for 'space'
    
    if action is not None:
        # Take a step in the environment with the selected action
        obs, reward, done, _, info = env.step(action)
        fig = obs["image"] # np array
        # repeat each pixel 4 times
        fig = np.repeat(fig, 4, axis=0)
        fig = np.repeat(fig, 4, axis=1)
        # Display the frame
        frame = pygame.surfarray.make_surface(fig)
        screen.blit(frame, (0, 0))
        total_reward += reward
        print(reward, total_reward)
    
    pygame.display.flip()  # Update the screen
    
    # Control the frame rate (you can adjust this based on your needs)
    clock.tick(10)

# Clean up
env.close()
pygame.quit()