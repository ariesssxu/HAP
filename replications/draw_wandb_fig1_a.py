import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

def time_weighted_ema(steps, rewards, span=20):
    mask = steps.notna() & rewards.notna()
    steps = steps[mask].to_numpy()
    rewards = rewards[mask].to_numpy()
    if len(rewards) == 0:
        return steps, rewards
    rewards_ema = pd.Series(rewards).ewm(span=span, adjust=False).mean().to_numpy()
    return steps, rewards_ema

df = pd.read_csv('wandb_export_2025-06-18T15_30_40.362+08_00.csv')
df = df.replace('', np.nan)
numeric_columns = df.columns[df.columns != 'step']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

colors = {
    'HAP (ours)': '#CE4C4A',
    'DQN (exp_frac=0.5)': '#5280C1', 
    'DQN (exp_frac=0.1)': '#1E9FA2', 
    'SAC': '#46945C',
    'A2C': '#E5B136',
    'TD3': '#AA4C9D'
}
algorithms = list(colors.keys())
total_step_col = 'step'

plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.set_facecolor('#FAFAFA')

for algo in algorithms:
    step_col = f'{algo} - _step'
    reward_col = f'{algo} - reward'
    if step_col in df.columns and reward_col in df.columns:
        valid = df[step_col].notna()
        steps = df.loc[valid, total_step_col]
        rewards = df.loc[valid, reward_col]

        ema_steps, ema_rewards = time_weighted_ema(steps, rewards, span=20)
        if len(ema_rewards) > 0:
            color = colors[algo]
            plt.plot(ema_steps, ema_rewards, linestyle='-', color=color, linewidth=3, alpha=0.9, zorder=10)


for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#333333')
    ax.spines[spine].set_linewidth(1.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis='x', colors='#333333', labelsize=12, width=1.5)
ax.tick_params(axis='y', colors='#333333', labelsize=12, width=1.5)

xticks = np.linspace(0, 200000, 9, dtype=int)  # [0, 25000, ..., 200000]
xticklabels = [f'{int(x/1000)}k' for x in xticks]  # ['0k', '25k', ..., '200k']

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
plt.ylabel('Reward', fontsize=14, fontweight='bold')
# plt.title('RL Algorithm Comparison', fontsize=16, pad=10)
plt.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0], [0], color=colors[algo], linewidth=3, label=algo)
    for algo in algorithms
]
# legend_elements.append(Line2D([0], [0], color='gray', alpha=0.3, linewidth=8, label='± Estimated Range'))

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.12),
    ncol=len(legend_elements),
    fontsize=10.5,
    # prop={'weight': 'bold'},
    frameon=False,
    columnspacing=0.5
)

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.15)

plt.savefig('rl_algorithm_comparison_final.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.close()
print("Final RL reward curve saved as rl_algorithm_comparison_final.pdf")
