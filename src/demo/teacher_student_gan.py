import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a Task class
class Task:
    def __init__(self, level, expected_reward):
        self.level = level
        self.expected_reward = expected_reward

# Create tasks with different levels and expected rewards
tasks = [
    Task(level=0, expected_reward=0.1),
    Task(level=1, expected_reward=0.2),
    Task(level=2, expected_reward=0.5),
    Task(level=3, expected_reward=0.7)
]

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

# Learner (Student) Neural Network
class TaskLearner(nn.Module):
    def __init__(self, n_tasks):
        super(TaskLearner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_tasks, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, task_one_hot):
        return self.model(task_one_hot)

# Initialize the networks and optimizers
n_tasks = len(tasks)
generator = TaskGenerator(n_tasks)
learner = TaskLearner(n_tasks)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.01)
learn_optimizer = optim.Adam(learner.parameters(), lr=0.01)

# Training setup
n_iterations = 1000
loss_function = nn.BCELoss()

# Metrics for analysis and visualization
gen_losses, learn_losses = [], []
task_prob_over_time, learner_success_rate = [], []

for iteration in range(n_iterations):
    # Generator selects a task
    task_idx, task_probs, log_prob = generator()
    selected_task = tasks[task_idx.item()]
    reward_prob = selected_task.expected_reward
    # Simulated reward based on the expected reward probability
    reward = 1 if np.random.rand() < reward_prob else 0

    # ---- Learner Update ----
    # Learner tries to solve the generated task
    task_one_hot = torch.zeros(n_tasks)
    task_one_hot[task_idx] = 1.0
    task_one_hot += torch.normal(mean=0.0, std=0.01, size=task_one_hot.size())
    task_one_hot = task_one_hot.unsqueeze(0)  # Add batch dimension

    # Forward pass for learner
    learner_prediction = learner(task_one_hot)
    learner_target = torch.tensor([[reward]], dtype=torch.float32)
    learn_loss = loss_function(learner_prediction, learner_target)

    # Backward pass and optimization for learner
    learn_optimizer.zero_grad()
    learn_loss.backward()
    learn_optimizer.step()

    # ---- Generator Update ----
    # Compute the generator's reward (1 when learner fails)
    reward_gen = 1 - reward  # Generator wants the learner to fail

    # Compute generator's loss using REINFORCE
    gen_loss = -log_prob * reward_gen  # Negative sign because we minimize loss

    # Backward pass and optimization for generator
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    # Track losses and metrics
    gen_losses.append(gen_loss.item())
    learn_losses.append(learn_loss.item())
    task_prob_over_time.append(task_probs.detach().numpy())
    learner_success_rate.append(reward)

    # Print progress every 100 iterations
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}: Generator Loss: {gen_loss.item():.4f}, Learner Loss: {learn_loss.item():.4f}")

# Convert task probabilities to NumPy array for plotting
task_prob_over_time = np.array(task_prob_over_time)

# Plotting
plt.figure(figsize=(15, 5))

# Plot Generator and Learner Loss over Training
plt.subplot(1, 3, 1)
plt.plot(gen_losses, label='Generator Loss')
plt.plot(learn_losses, label='Learner Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Generator and Learner Loss over Training')
plt.legend()
plt.grid(True)

# Plot Learner Success Rate Over Time
plt.subplot(1, 3, 2)
window_size = 100
success_rate_smoothed = np.convolve(learner_success_rate, np.ones(window_size)/window_size, mode='valid')
plt.plot(success_rate_smoothed, color='green')
plt.xlabel('Iterations')
plt.ylabel('Success Rate')
plt.title('Learner Success Rate Over Time')
plt.grid(True)

# Plot Task Selection Probabilities Over Time
plt.subplot(1, 3, 3)
for i in range(n_tasks):
    plt.plot(task_prob_over_time[:, i], label=f'Task Level {tasks[i].level}')
plt.title('Task Selection Probabilities Over Time')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("teacher_student_gan.png")
plt.show()