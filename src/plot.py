import matplotlib.pyplot as plt
import numpy as np
import json
import plot_utils as pu

RESAMPLE = 1000
SMOOTH=5

# Define the plotting function with conditional checks for empty inputs
def plot_results(episode_rewards=None, success_rates_list=None, task_indices_over_time=None, task_prob_over_time=None,
                 task_success_rates_over_time=None, eval_success_rates_over_time=None,
                 eval_steps=None, n_tasks=4, tasks=None, save_path="teacher_student_gan_rl_with_teacher.png"):
    """
    Plots various training metrics. Each subplot is only generated if the corresponding data is provided.

    Parameters:
    - episode_rewards (list): List of rewards per episode.
    - success_rates_list (list): List of success flags per episode.
    - task_indices_over_time (list): List of task indices selected over time.
    - task_success_rates_over_time (list of np.arrays): Success rates per task over evaluation steps.
    - eval_success_rates_over_time (list of lists): Evaluation success rates per task over time.
    - eval_steps (list): List of episode numbers where evaluations were performed.
    - n_tasks (int): Number of tasks.
    - tasks (list): List of tasks (for labeling).
    - save_path (str): Path to save the generated plot.
    """

    num_plots = 0
    if episode_rewards and len(episode_rewards) > 0:
        num_plots += 1
    if success_rates_list and len(success_rates_list) > 0:
        num_plots += 1
    if task_indices_over_time and len(task_indices_over_time) > 0:
        num_plots += 1
    if task_success_rates_over_time and len(task_success_rates_over_time) > 0:
        num_plots += 1
    if eval_success_rates_over_time and len(eval_success_rates_over_time) > 0:
        num_plots += 1
    if task_prob_over_time and len(task_prob_over_time) > 0:
        num_plots += 1

    if num_plots == 0:
        print("No data provided for plotting.")
        return

    plt.figure(figsize=(5 * num_plots, 5))

    plot_idx = 1

    # Plot Episode Rewards
    if episode_rewards and len(episode_rewards) > 0:
        ax = plt.subplot(1, num_plots, plot_idx)
        x = np.arange(len(episode_rewards))
        y = np.array(episode_rewards)
        pu.plot_results(
            x,
            y,
            ax=ax,
            title='Episode Rewards over Time',
            legend_label='Rewards',
            shaded_std=True,
            shaded_err=True,
            resample=RESAMPLE,
            smooth_step=SMOOTH,
        )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.grid(True)
        plot_idx += 1

    # Plot Success Rate Over Time
    if success_rates_list and len(success_rates_list) > 0:
        ax = plt.subplot(1, num_plots, plot_idx)
        if len(success_rates_list) >= 100:
            success_rate_smoothed = np.convolve(success_rates_list, np.ones(100)/100, mode='valid')
            x = np.arange(len(success_rate_smoothed))
            y = np.array(success_rate_smoothed)
        else:
            x = np.arange(len(success_rates_list))
            y = np.array(success_rates_list)
        pu.plot_results(
            x,
            y,
            ax=ax,
            title='Success Rate Over Time',
            legend_label='Success Rate',
            shaded_std=True,
            shaded_err=True,
            resample=RESAMPLE,
            smooth_step=SMOOTH,
        )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate')
        ax.grid(True)
        plot_idx += 1

    # Plot Task Distribution Over Time
    if task_indices_over_time and len(task_indices_over_time) > 0:
        ax = plt.subplot(1, num_plots, plot_idx)
        task_indices_array = np.array(task_indices_over_time)
        for i in range(n_tasks):
            task_counts = np.cumsum(task_indices_array == i)
            x = np.arange(len(task_counts))
            y = task_counts
            pu.plot_results(
                x,
                y,
                ax=ax,
                title='Cumulative Task Counts Over Time',
                legend_label=f'Task {i} (len={len(tasks[i])})',
                resample=RESAMPLE,
                smooth_step=SMOOTH,
            )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Cumulative Count')
        ax.legend()
        ax.grid(True)
        plot_idx += 1
        
    # plot task_prob_over_time for each task
    if task_prob_over_time and len(task_prob_over_time) > 0:
        ax = plt.subplot(1, num_plots, plot_idx)
        task_prob_array = np.array(task_prob_over_time)
        for i in range(n_tasks):
            x = np.arange(len(task_prob_array))
            y = task_prob_array[:, i]
            pu.plot_results(
                x,
                y,
                ax=ax,
                title='Task Probabilities Over Time',
                legend_label=f'Task {i} (len={len(tasks[i])})',
                resample=RESAMPLE,
                smooth_step=SMOOTH,
            )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Task Probability')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot success rate over time for each task
    if task_success_rates_over_time and len(task_success_rates_over_time) > 0 and eval_steps:
        ax = plt.subplot(1, num_plots, plot_idx)
        success_rates_over_time = np.array(task_success_rates_over_time)
        for task_id in range(n_tasks):
            x = eval_steps
            y = success_rates_over_time[:, task_id]
            pu.plot_results(
                x,
                y,
                ax=ax,
                title='Success Rate Over Time for Each Task',
                legend_label=f'Task {task_id}',
                shaded_std=True,
                shaded_err=True,
                resample=RESAMPLE,
                smooth_step=SMOOTH,
            )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot Evaluation Success Rates Over Time
    if eval_success_rates_over_time and len(eval_success_rates_over_time) > 0 and eval_steps:
        ax = plt.subplot(1, num_plots, plot_idx)
        eval_success_rates_over_time = np.array(eval_success_rates_over_time)
        for task_id in range(n_tasks):
            x = eval_steps
            y = eval_success_rates_over_time[:, task_id]
            pu.plot_results(
                x,
                y,
                ax=ax,
                title='Evaluation Success Rates Over Time',
                legend_label=f'Task {task_id} (len={len(tasks[task_id])})',
                shaded_std=True,
                shaded_err=True,
                resample=RESAMPLE,
                smooth_step=SMOOTH,
            )
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plots saved to {save_path}")

# Plotting function
def plot_success_rates(success_rates=None, eval_steps=None, tasks=None, save_path="success_rates.png"):
    """
    Plots the success rates per task over time. Handles empty inputs gracefully.

    Parameters:
    - success_rates (dict): Dictionary of success rates per task.
    - eval_steps (list): List of timesteps where evaluations were performed.
    - tasks (list): List of task names.
    - save_path (str): Path to save the generated plot.
    """
    if not success_rates or not eval_steps or not tasks:
        print("Insufficient data provided for plotting success rates.")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for task_name in tasks:
        rates = success_rates[task_name]
        if rates:
            x = eval_steps
            y = rates
            pu.plot_results(
                x,
                y,
                ax=ax,
                title='Success Rates per Task Over Time',
                legend_label=task_name,
                shaded_std=True,
                shaded_err=True,
                resample=RESAMPLE,
                smooth_step=SMOOTH,
            )

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Success Rate')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Success rate plot saved to {save_path}")

def plot_results_from_files(result_files, labels, save_path):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    for result_file, label in zip(result_files, labels):
        with open(result_file, 'r') as f:
            results = json.load(f)

        episode_rewards = results['episode_rewards']
        eval_results = results['eval_results']
        eval_steps = [res['step'] for res in eval_results]
        eval_success_rates_over_time = [res['success_rates'] for res in eval_results]

        # Plot episode rewards
        x_rewards = np.arange(len(episode_rewards))
        y_rewards = episode_rewards
        pu.plot_results(
            x_rewards,
            y_rewards,
            ax=ax,
            title='Training Performance Comparison',
            legend_label=f'{label} - Rewards',
            shaded_std=True,
            shaded_err=True,
            resample=RESAMPLE,
            smooth_step=SMOOTH,
        )

        # Plot evaluation success rates
        # Here we plot the average success rate across tasks at each evaluation
        avg_success_rates = [np.mean(srs) for srs in eval_success_rates_over_time]
        x_eval = eval_steps
        y_eval = avg_success_rates
        pu.plot_results(
            x_eval,
            y_eval,
            ax=ax,
            title='Training Performance Comparison',
            legend_label=f'{label} - Eval Success Rate',
            shaded_std=True,
            shaded_err=True,
            resample=RESAMPLE,
            smooth_step=SMOOTH,
        )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward / Success Rate')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path)
    plt.show()