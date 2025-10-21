import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the Teacher class (ensure this is already imported or defined in your environment)

class Teacher(object):
    """Teacher using the Exponential-weight algorithm for Exploration and Exploitation (Exp3) algorithm."""

    def __init__(self, tasks, gamma=0.3):
        self._tasks = tasks
        self._n_tasks = len(self._tasks)
        self._gamma = gamma
        self._log_weights = torch.zeros(self._n_tasks, dtype=torch.float32)

    @property
    def task_probabilities(self):
        weights = torch.exp(self._log_weights - torch.sum(self._log_weights))
        probs = (1 - self._gamma) * weights / torch.sum(weights) + self._gamma / self._n_tasks
        return probs

    def get_task(self):
        """Samples a task according to the current Exp3 belief."""
        task_probs = self.task_probabilities
        task_distribution = torch.distributions.Categorical(probs=task_probs)
        task_i = task_distribution.sample().item()
        return self._tasks[task_i]

    def update(self, task, reward):
        task_i = self._tasks.index(task)
        task_probs = self.task_probabilities

        reward_corrected = reward / task_probs[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks

# Define tasks and expected rewards
tasks = [0, 1, 2, 3]
expected_rewards = [0.1, 0.2, 0.5, 0.7]  # Expected rewards for each task

# Instantiate the teacher
teacher = Teacher(tasks=tasks, gamma=0.3)

# Run the simulation
n_iterations = 1000
task_probabilities_over_time = []
selected_tasks = []
rewards = []

for t in range(n_iterations):
    # Teacher selects a task
    task = teacher.get_task()
    task_index = tasks.index(task)
    
    # Simulate the reward for the selected task
    reward_prob = expected_rewards[task_index]
    reward = 1 if np.random.rand() < reward_prob else 0
    
    # Teacher updates beliefs based on the received reward
    teacher.update(task, reward)
    
    # Record data
    task_probabilities_over_time.append(teacher.task_probabilities.clone().detach().numpy())
    selected_tasks.append(task)
    rewards.append(reward)

# Convert recorded probabilities to a NumPy array for easier handling
task_probabilities_over_time = np.array(task_probabilities_over_time)

# Plot the task probabilities over time
plt.figure(figsize=(12, 6))
for i, task in enumerate(tasks):
    plt.plot(task_probabilities_over_time[:, i], label=f'Task {task}')
plt.title('Task Selection Probabilities Over Time')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("test.png")