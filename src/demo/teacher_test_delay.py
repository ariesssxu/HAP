import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the Teacher class (ensure this is already imported or defined in your environment)

class Teacher(object):
    """Teacher using the Exponential-weight algorithm for Exploration and Exploitation (Exp3) algorithm with delayed rewards."""

    def __init__(self, tasks, gamma=0.3):
        self._tasks = tasks
        self._n_tasks = len(self._tasks)
        self._gamma = gamma
        self._log_weights = torch.zeros(self._n_tasks, dtype=torch.float32)
    
    @property
    def task_probabilities(self):
        weights = torch.exp(self._log_weights - torch.max(self._log_weights))  # Subtract max for numerical stability
        probs = (1 - self._gamma) * weights / torch.sum(weights) + self._gamma / self._n_tasks
        return probs

    def get_task(self):
        """Samples a task according to the current Exp3 belief."""
        task_probs = self.task_probabilities
        task_distribution = torch.distributions.Categorical(probs=task_probs)
        task_i = task_distribution.sample().item()
        return self._tasks[task_i], task_probs[task_i].item()  # Return the task and its probability at selection time

    def update(self, task_index, reward, task_prob_at_selection):
        reward_corrected = reward / task_prob_at_selection
        self._log_weights[task_index] += self._gamma * reward_corrected / self._n_tasks

# Define tasks and expected rewards
tasks = [0, 1, 2, 3]
expected_rewards = [0.1, 0.2, 0.5, 0.7]  # Expected rewards for each task

# Instantiate the teacher
teacher = Teacher(tasks=tasks, gamma=0.3)

# Simulation parameters
n_iterations = 1000
delay = 50  # Delay in iterations before the reward is received

# Data storage
task_probabilities_over_time = []
selected_tasks = []
rewards = []
pending_rewards = []

for t in range(n_iterations):
    # Teacher selects a task
    task, task_prob_at_selection = teacher.get_task()
    task_index = tasks.index(task)
    
    # Simulate the reward for the selected task (to be received after 'delay' iterations)
    reward_prob = expected_rewards[task_index]
    reward = 1 if np.random.rand() < reward_prob else 0
    
    # Store the pending reward to be processed after 'delay' iterations
    receive_time = t + delay
    pending_rewards.append({
        'receive_time': receive_time,
        'task_index': task_index,
        'reward': reward,
        'task_prob_at_selection': task_prob_at_selection
    })
    
    # Process any rewards that are due at the current time
    rewards_to_process = [pr for pr in pending_rewards if pr['receive_time'] == t]
    for pr in rewards_to_process:
        teacher.update(pr['task_index'], pr['reward'], pr['task_prob_at_selection'])
        rewards.append(pr['reward'])
        # Remove the processed reward from pending_rewards
        pending_rewards.remove(pr)
    
    # Record data
    task_probabilities_over_time.append(teacher.task_probabilities.clone().detach().numpy())
    selected_tasks.append(task)
    
    # Optional: To handle cases where no rewards are processed in the first 'delay' iterations
    if t < delay:
        rewards.append(0)  # Append 0 rewards for initial steps

# Convert recorded probabilities to a NumPy array for easier handling
task_probabilities_over_time = np.array(task_probabilities_over_time)

# Plot the task probabilities over time
plt.figure(figsize=(12, 6))
for i, task in enumerate(tasks):
    plt.plot(task_probabilities_over_time[:, i], label=f'Task {task}')
plt.title('Task Selection Probabilities Over Time with Delayed Rewards')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("test_delay.png")
