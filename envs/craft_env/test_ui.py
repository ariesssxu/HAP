import pygame
import gym
import numpy as np
import os

# Import your environment, assuming CraftEnv is the Gym environment.
from env_gym import CraftEnv

# Initialize pygame
pygame.init()

# Set up the window size (you can adjust this based on your needs)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Human-Interactive Gym Environment")

# Set up the clock to control the frame rate
clock = pygame.time.Clock()

# Create the environment (replace with your own environment paths)
local_path = os.path.dirname(os.path.realpath(__file__))
recipes_path = os.path.join(local_path, "resources", "recipes.yaml")
hints_path = os.path.join(local_path, "resources", "hints.yaml")
env = CraftEnv(recipes_path, hints_path, visualise=True, max_steps=1000, accumulate_reward=True)

# Reset the environment
obs = env.reset()

done = False

total_reward = 0

print(env.env.world.cookbook.index)
print(env.env._current_state.inventory)

preset_action_sequence = []

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Handling key events for movement
    keys = pygame.key.get_pressed()
    action = None  # Define action here based on key press
    
    if keys[pygame.K_w]:  # Move forward (you can adjust this)
        action = 0  # Replace with actual action index for 'w'
    elif keys[pygame.K_s]:  # Move backward
        action = 1  # Replace with actual action index for 's'
    elif keys[pygame.K_a]:  # Move left
        action = 2  # Replace with actual action index for 'a'
    elif keys[pygame.K_d]:  # Move right
        action = 3  # Replace with actual action index for 'd'
    elif keys[pygame.K_SPACE]:  # Interact (e.g., select)
        action = 4  # Replace with actual action index for 'space'
    
    if action is not None:
        # Take a step in the environment with the selected action
        obs, reward, done, info = env.step(action)
        fig = env.render()
        print(env.env._current_state.inventory)
        total_reward += reward
        print(reward, total_reward)
    
    pygame.display.flip()  # Update the screen
    
    # Control the frame rate (you can adjust this based on your needs)
    clock.tick(30)

# Clean up
env.close()
pygame.quit()