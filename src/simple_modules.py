import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import json
# from simple_env import *

# Generator (Teacher) Neural Network
class TaskGenerator(nn.Module):
    def __init__(self, n_tasks):
        super(TaskGenerator, self).__init__()
        # Learnable logits for task probabilities
        self.logits = nn.Parameter(torch.randn(n_tasks))
        
    def forward(self):
        task_probs = torch.softmax(self.logits, dim=0)
        task_distribution = torch.distributions.Categorical(task_probs)
        task_idx = task_distribution.sample()
        log_prob = task_distribution.log_prob(task_idx)
        return task_idx, task_probs, log_prob

    def reset(self, probs=None):
        if probs is not None:
            # Convert provided probabilities to logits (inverse of softmax)
            with torch.no_grad():
                self.logits.copy_(torch.log(torch.tensor(probs)))
        else:
            # If no probabilities are provided, reset to default initialization
            with torch.no_grad():
                self.logits.copy_(torch.randn_like(self.logits))
                
# Actor-Critic Networks
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        out = F.relu(out)
        return out

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_actions=2):
        super(ActorNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            # ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.res_blocks(x)
        logits = self.fc_out(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            # ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.res_blocks(x)
        value = self.fc_out(x)
        return value

# for logging
class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, eval_env, eval_freq=1000, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.eval_results = []
        self.save_path = os.path.join(log_dir, 'results.json')
    
    def _on_step(self) -> bool:
        # Record the episode reward
        if self.locals.get('done', False):
            episode_reward = self.locals.get('reward')
            self.episode_rewards.append(episode_reward)
        
        # Evaluate the model every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            success_rates = evaluate_stable_baselines_model(self.model, self.eval_env, n_eval_episodes=30)
            self.eval_results.append({
                'step': self.n_calls,
                'success_rates': success_rates
            })
            # Optionally print or log the results
            print(f"Step {self.n_calls}: Success Rates: {success_rates}")
        
        return True  # Return True to continue training
    
    def _on_training_end(self):
        # Save the results to a file
        results = {
            'episode_rewards': self.episode_rewards,
            'eval_results': self.eval_results
        }
        with open(self.save_path, 'w') as f:
            json.dump(results, f)

def evaluate_stable_baselines_model(model, env, n_eval_episodes=30):
    success_rates = []
    for task_id in range(n_tasks):
        successes = 0
        env.reset_task(task_id)
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    if reward >= expected_rewards[task_id]:
                        successes += 1
                    break
        success_rate = successes / n_eval_episodes
        success_rates.append(success_rate)
    return success_rates