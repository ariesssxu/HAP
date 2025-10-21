import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from simple_modules import *
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import DQN, A2C  # DDPG is not suitable for discrete action spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import json
from simple_env import TaskEnv, task_sequences, expected_rewards, n_tasks, n_actions, tasks
# Import wandb
import wandb
# Call the plotting function
from plot import *

# Initialize wandb
wandb.init(project='teacher_student_gan_rl', name='training_run')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

gamma = 0.999  # Discount factor

def main(args):

    # Initialize the environment
    log_dir = os.path.join(args.log_dir, args.algo)
    env = TaskEnv(task_sequences, expected_rewards)
    env.reset_task(0)
    env = Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))
    

    if args.algo == 'default':

        # Initialize TaskGenerator and its optimizer
        task_generator = TaskGenerator(n_tasks)
        task_generator.reset(probs=[0.7, 0.1, 0.1, 0.1])  # Uniform probabilities
        task_optimizer = optim.Adam(task_generator.parameters(), lr=5e-2)

        # Initialize the environment with the task sequences and expected rewards

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = env.max_steps + n_tasks  # 16 + 4 = 20

        actor_net = ActorNet(input_dim, hidden_dim=128, n_actions=n_actions).to(device)
        value_net = ValueNet(input_dim, hidden_dim=128).to(device)

        optimizer = optim.Adam(list(actor_net.parameters()) + list(value_net.parameters()), lr=2e-5, betas=(0.9, 0.999))

        # Function to select action based on action probabilities
        def select_action(action_probs):
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            return action.item(), log_prob

        # Evaluation function
        def evaluate(policy_net, env, n_eval_episodes=30):
            # Evaluate the agent's performance on each task
            success_rates = []
            for task_id in range(n_tasks):
                successes = 0
                for i in range(n_eval_episodes):
                    env.reset_task(task_id)
                    observation = env.reset()

                    done = False
                    while not done:
                        # Prepare the input for the policy network
                        current_step_input = torch.tensor(observation['current_step'], dtype=torch.float32).to(device)
                        task_flag = torch.tensor(observation['task_flag'], dtype=torch.float32).to(device)
                        current_input = torch.cat([current_step_input, task_flag], dim=0).unsqueeze(0)  # batch size 1

                        with torch.no_grad():
                            action_probs = policy_net(current_input)

                        # Select action greedily (max probability)
                        action_idx = torch.argmax(action_probs).item()

                        # Step the environment
                        observation, reward, done, info = env.step(action_idx)

                        if done:
                            if reward >= expected_rewards[task_id]:
                                successes += 1
                            break
                success_rate = successes / n_eval_episodes
                success_rates.append(success_rate)
            return success_rates

        # Training Loop
        n_episodes = 100000
        gamma = 0.99
        teacher_update_freq = 50

        episode_rewards = []
        task_indices_over_time = []
        success_rates_list = []
        task_success_rates = np.zeros(n_tasks)

        # For moving average of success rates per task
        success_histories = [deque(maxlen=100) for _ in range(n_tasks)]  # For smoothing

        eval_interval = 1000
        eval_success_rates_over_time = []
        task_success_rates_over_time = []
        task_prob_over_time = []
        eval_steps = []

        for episode in range(n_episodes):
            # Get task index from TaskGenerator
            task_idx, task_probs, log_prob_task = task_generator()
            task_idx = task_idx.item()
            expected_reward = expected_rewards[task_idx]

            # Set the task in the environment
            env.reset_task(task_idx)

            # Reset the environment
            observation = env.reset()

            log_probs = []
            values = []
            rewards = []
            total_reward = 0.0

            done = False
            t = 0
            while not done:
                # Prepare the input for the policy network
                current_step_input = torch.tensor(observation['current_step'], dtype=torch.float32).to(device)
                task_flag = torch.tensor(observation['task_flag'], dtype=torch.float32).to(device)
                current_input = torch.cat([current_step_input, task_flag], dim=0).unsqueeze(0)  # batch size 1

                # Get action probabilities from actor network
                action_probs = actor_net(current_input)
                value = value_net(current_input)

                # Sample action
                action_idx, log_prob = select_action(action_probs)

                # Step the environment
                observation, reward, done, info = env.step(action_idx)

                # Save log prob and value for training
                log_probs.append(log_prob)
                values.append(value.squeeze())
                rewards.append(reward)
                total_reward += reward

                t += 1

            wandb.log({'episode_reward': total_reward})

            # Compute returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            values = torch.stack(values)

            advantages = returns - values.detach()

            # Compute actor and critic losses
            log_probs = torch.stack(log_probs)
            actor_loss = - (log_probs * advantages).sum()
            critic_loss = advantages.pow(2).sum()
            loss = actor_loss + critic_loss

            # Optimize the actor and critic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update success history for the task
            success = 1 if total_reward >= expected_reward else 0
            success_histories[task_idx].append(success)
            task_success_rates[task_idx] = np.mean(success_histories[task_idx]) if len(success_histories[task_idx]) > 0 else 0.0

            # Compute the teacher's loss
            # The teacher aims to sample tasks where the agent's success rate is low
            # So we define the teacher's loss to encourage sampling tasks with lower success rates

            # Convert success rates to tensor
            success_rates_tensor = torch.tensor(task_success_rates, dtype=torch.float32).to(device)

            # Compute the entropy of the task probabilities
            entropy = -torch.sum(task_probs * torch.log(task_probs + 1e-8))

            # Set the entropy regularization coefficient
            entropy_coeff = 0.1  # Adjust this value as needed

            
            # Compute teacher's loss
            # We want to maximize the negative of success rates (i.e., minimize success rates)
            # So the loss is the dot product of task probabilities and success rates
            # Since we are minimizing, we multiply by -1

            # teacher_loss = torch.dot(task_probs.to(device), success_rates_tensor)

            # Design 1
            # The teacher aims to minimize the expected gain of the agent
            # _, _, log_prob_task = task_generator()
            # teacher_loss = -log_prob_task * total_reward  # Negative sign to minimize agent's reward

            # Design 2
            teacher_loss = torch.dot(task_probs.to(device), success_rates_tensor) - entropy_coeff * entropy
            # teacher_loss = torch.dot(task_probs.to(device), success_rates_tensor) - entropy_coeff * entropy

            # Update the teacher every N steps
            if (episode + 1) % teacher_update_freq == 0:
                task_optimizer.zero_grad()
                teacher_loss.backward()
                task_optimizer.step()

            # Store metrics
            episode_rewards.append(total_reward)
            task_indices_over_time.append(task_idx)
            success_rates_list.append(success)

            # Print Teacher's task probabilities
            with torch.no_grad():
                task_probs = torch.softmax(task_generator.logits, dim=0)
                # print(f"Teacher's Task Probabilities: {task_probs.cpu().numpy()}")
                task_prob_over_time.append(task_probs.cpu().numpy())
                wandb.log({f'task_prob_task_{i}': task_probs[i].item() for i in range(n_tasks)})

            # Log metrics to wandb
            wandb.log({
                'episode': episode + 1,
                'reward': total_reward,
                'task_idx': task_idx,
                'success': success,
                'loss': loss.item(),
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                f'success_rate_task_{task_idx}': task_success_rates[task_idx]
            })

            # Evaluate every eval_interval episodes
            if (episode + 1) % eval_interval == 0:
                task_success_rates_over_time.append(task_success_rates.copy())
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                avg_success = np.mean(success_rates_list[-eval_interval:])
                print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, Avg Success Rate: {avg_success:.2f}")
                wandb.log({'avg_reward': avg_reward, 'avg_success_rate': avg_success})

                # Evaluate the agent
                success_rates_per_task = evaluate(actor_net, env, n_eval_episodes=30)
                eval_success_rates_over_time.append(success_rates_per_task)
                eval_steps.append(episode + 1)
                print(f"Evaluation Success Rates (per task): {success_rates_per_task}")
                wandb.log({f'eval_success_rate_task_{i}': success_rates_per_task[i] for i in range(n_tasks)})

                print(f"Teacher's Task Probabilities: {task_probs.cpu().numpy()}")

                # Log evaluation metrics to wandb
                eval_metrics = {f'eval_success_rate_task_{i}': success_rates_per_task[i] for i in range(n_tasks)}
                wandb.log(eval_metrics)

        plot_results(
            episode_rewards=episode_rewards,
            success_rates_list=success_rates_list,
            task_indices_over_time=task_indices_over_time,
            task_prob_over_time=task_prob_over_time,
            task_success_rates_over_time=task_success_rates_over_time,
            eval_success_rates_over_time=eval_success_rates_over_time,
            eval_steps=eval_steps,
            n_tasks=n_tasks,
            tasks=tasks,
            save_path="teacher_student_gan_rl_with_teacher.png"
        )
    
    else: 
            
        # For demonstration, we'll train on task 0
        env.reset_task(0)

        eval_env = TaskEnv(task_sequences, expected_rewards)
        eval_env.reset_task(0)
        eval_env = Monitor(eval_env, filename=os.path.join(log_dir, 'eval_monitor.csv'))

        net_arch = [128]
        policy_kwargs = dict(net_arch=net_arch)

        # Create a Stable Baselines3 model based on the chosen algorithm
        if args.algo == 'dqn':
            model = DQN('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
        elif args.algo == 'a2c':
            model = A2C('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            print("Invalid algorithm selected or algorithm not supported.")
            exit(1)

        # Evaluate the model during training
        custom_callback = CustomLoggingCallback(log_dir=log_dir, eval_env=eval_env, eval_freq=1000)

        # Train the model
        total_timesteps = 300000  # Adjust as needed
        model.learn(total_timesteps=total_timesteps, callback=custom_callback)

        # Save the model
        model.save(f"{args.algo}_model")

        # Evaluate the model after training
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        
        # After training, you can plot the results
        # For example, if you have trained with different algorithms and saved their results
        result_files = [
            # os.path.join('./logs/default', 'results_default.json'),
            os.path.join('./logs/dqn', 'results.json'),
            os.path.join('./logs/a2c', 'results.json')
        ]
        labels = ['Default Algorithm', 'DQN', 'A2C']

        plot_results_from_files(result_files, labels, save_path='comparison_plot.png')

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Choose the learning algorithm.")
    parser.add_argument('--algo', type=str, default='default', choices=['default', 'dqn', 'a2c'],
                        help='Learning algorithm to use: default, dqn, a2c')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory to save logs and results.')
    args = parser.parse_args()
    main(args)