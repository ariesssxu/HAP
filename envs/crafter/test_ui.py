import pygame
import gym
import numpy as np
import os
import imageio
import shutil
from copy import deepcopy
import crafter

# Initialize Pygame
pygame.init()

# Set up the window
WINDOW_WIDTH = 512
WINDOW_HEIGHT = 512
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Human-Interactive Gym Environment")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Create the environment
env = crafter.Env(length=1000000, invincible=True)
obs = env.reset()

# Prepare the frames directory
if os.path.exists('frames'):
    shutil.rmtree('frames')
os.makedirs('frames', exist_ok=False)
frame_count = 0

# Game loop
done = False
total_reward = 0

preset_action_sequence = []
frames = []

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    action = None

    # Map keys to actions
    if keys[pygame.K_w]:
        action = 3  # Move forward
    elif keys[pygame.K_s]:
        action = 4  # Move backward
    elif keys[pygame.K_a]:
        action = 1  # Move left
    elif keys[pygame.K_d]:
        action = 2  # Move right
    elif keys[pygame.K_SPACE]:
        action = 5  # Interact
    elif keys[pygame.K_1]:
        action = 6
    elif keys[pygame.K_2]:
        action = 7
    elif keys[pygame.K_3]:
        action = 8
    elif keys[pygame.K_4]:
        action = 9
    elif keys[pygame.K_5]:
        action = 10
    elif keys[pygame.K_6]:
        action = 11
    elif keys[pygame.K_7]:
        action = 12
    elif keys[pygame.K_8]:
        action = 13
    elif keys[pygame.K_9]:
        action = 14
    elif keys[pygame.K_0]:
        action = 15
    elif keys[pygame.K_p]:
        action = 16
    elif keys[pygame.K_ESCAPE]:
        done = True

    if action is not None:
        # Step the environment
        obs, reward, done, info = env.step(action)
        fig = env.render()  # Get the observation image

        fig = np.repeat(fig, 8, axis=0)
        fig = np.repeat(fig, 8, axis=1)

        frames.append(deepcopy(fig))

        # Process the image
        fig = np.rot90(fig)
        fig = fig[::-1, :]

        # Convert to surface and display
        frame = pygame.surfarray.make_surface(fig)
        screen.blit(frame, (0, 0))
        total_reward += reward
        print(f"Reward: {reward}, Total: {total_reward}")

        filename = os.path.join('frames', f'frame_{frame_count:04d}.png')
        frame_count += 1

    pygame.display.flip()
    clock.tick(10)  # 10 FPS

# Save the GIF
imageio.mimsave('episode.gif', frames, duration=0.1)  # 0.1 sec per frame = 10 FPS

# Clean up
env.close()
pygame.quit()
