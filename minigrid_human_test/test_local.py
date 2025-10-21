import pygame
import numpy as np
import gym
import imageio
from minigrid_human_test.envs import *
from minigrid_human_test.wrappers import RGBImgObsWrapper
import random
import os

# Configuration Constants
WINDOW_SIZE = (512, 512)
FRAME_RATE = 10
ENV_INDICES = [0, 1, 2, 3, 4, 5]  # All six environments

# Key to action mapping
KEY_ACTIONS = {
    pygame.K_a: 0,    # Move forward
    pygame.K_d: 1,    # Turn left
    pygame.K_w: 2,    # Move backward
    pygame.K_s: 7,    # Custom action 1
    pygame.K_o: 3,    # Turn right
    pygame.K_p: 4,    # Custom action 2
    pygame.K_l: 5,    # Custom action 3
    pygame.K_ESCAPE: 6,  # Toggle/interact
}

def initialize_environment(env_index):
    """Create and wrap the MiniGrid environment"""
    tasks = [
        EmptyEnv(max_steps=20, agent_start_pos=(random.randint(1, 3), random.randint(1, 3))),
        CrossingEnv(max_steps=40),
        DoorKeyEnv(max_steps=40),
        FourRoomsEnv(max_steps=100),
        LockedRoomEnv(max_steps=200),
        PlaygroundEnv(max_steps=200)
    ]
    return RGBImgObsWrapper(tasks[env_index])

def process_input():
    """Handle keyboard input and return corresponding action"""
    keys = pygame.key.get_pressed()
    for key, action in KEY_ACTIONS.items():
        if keys[key]:
            return action
    return None

def process_observation(observation):
    """Convert observation to numpy array for GIF (same as screen rendering)"""
    img_array = np.transpose(observation["image"], (1, 0, 2))  # (width, height, 3)
    surf = pygame.surfarray.make_surface(img_array)
    scaled_surf = pygame.transform.scale(surf, WINDOW_SIZE)
    frame_array = pygame.surfarray.array3d(scaled_surf)  # (width, height, 3)
    frame_array = np.transpose(frame_array, (1, 0, 2))  # (height, width, 3)
    return frame_array

def render_frame(screen, observation):
    """Render the environment observation to the screen"""
    img_array = np.transpose(observation["image"], (1, 0, 2))
    surf = pygame.surfarray.make_surface(img_array)
    scaled_surf = pygame.transform.scale(surf, WINDOW_SIZE)
    screen.blit(scaled_surf, (0, 0))

def main():
    """Main application loop with all environments in one GIF"""

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Human-Interactive MiniGrid - One GIF for All Environments")
    clock = pygame.time.Clock()

    # List to collect all frames across environments
    frames = []

    # Loop through each environment
    for env_index in ENV_INDICES:
        print(f"\nStarting Environment {env_index}...")

        # Initialize environment
        env = initialize_environment(env_index)
        obs = env.reset()[0]
        total_reward = 0

        # Save the initial frame of this environment
        frames.append(process_observation(obs))

        # Environment loop
        running_env = True
        done = False
        truncated = False
        while running_env and not done and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("User quit the program. Saving collected frames to GIF...")
                    running_env = False
                    done = True

            # Process user input
            action = process_input()

            if action is not None:
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                print(f"[Env {env_index}] Reward: {reward}, Total: {total_reward}")

                # Save the new frame
                frames.append(process_observation(obs))

            # Render to screen
            render_frame(screen, obs)
            pygame.display.flip()
            clock.tick(FRAME_RATE)

        env.close()

    # Save all collected frames to a single GIF
    os.makedirs('gifs', exist_ok=True)
    gif_path = 'gifs/episode_all.gif'
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"All environments completed. GIF saved to: {gif_path}")

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()
