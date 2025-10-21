import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set to not display windows
plt.ioff()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16


df = pd.read_csv('wandb_export_2025-06-17T14_19_16.286+08_00.csv')

# Convert Step to numeric
df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
for col in df.columns:
    if col != 'Step':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Select step range
df = df[(df['Step'] >= 12000) & (df['Step'] <= 20000)]

# Colors and task labels
n_tasks = 4
colors = ['#CE4C4A', '#5280C1', '#46945C', '#E5B136']
task_labels = ['Simple', 'Middle', 'Hard', 'Extremely Hard']

# Create smaller figure with narrow height and larger font size
plt.figure(figsize=(9, 3))

# Plot task probabilities and success rates with quantiles
for i in range(n_tasks):
    prob_col = f'training_run - task_prob_task_{i}'
    prob_min_col = f'training_run - task_prob_task_{i}__MIN'
    prob_max_col = f'training_run - task_prob_task_{i}__MAX'
    
    sr_col = f'training_run - success_rate_task_{i}'
    sr_min_col = f'training_run - success_rate_task_{i}__MIN'
    sr_max_col = f'training_run - success_rate_task_{i}__MAX'
    
    # Plot probability lines (dashed) with confidence intervals
    valid_idx_prob = df[prob_col].notna()
    if valid_idx_prob.any():
        steps_prob = df.loc[valid_idx_prob, 'Step']
        prob_values = df.loc[valid_idx_prob, prob_col]
        plt.plot(steps_prob, prob_values, linestyle='--', color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'{task_labels[i]} (Prob)')
        
        # Add confidence interval for probabilities if available
        if prob_min_col in df.columns and prob_max_col in df.columns:
            prob_min = df.loc[valid_idx_prob, prob_min_col]
            prob_max = df.loc[valid_idx_prob, prob_max_col]
            plt.fill_between(steps_prob, prob_min, prob_max, 
                           color=colors[i], alpha=0.1)
    
    # Plot success rate lines (solid) with confidence intervals
    valid_idx_sr = df[sr_col].notna()
    if valid_idx_sr.any():
        steps_sr = df.loc[valid_idx_sr, 'Step']
        sr_values = df.loc[valid_idx_sr, sr_col]
        plt.plot(steps_sr, sr_values, linestyle='-', color=colors[i], 
                linewidth=2, label=f'{task_labels[i]} (Success)')
        
        # Add confidence interval for success rates if available
        if sr_min_col in df.columns and sr_max_col in df.columns:
            sr_min = df.loc[valid_idx_sr, sr_min_col]
            sr_max = df.loc[valid_idx_sr, sr_max_col]
            plt.fill_between(steps_sr, sr_min, sr_max, 
                           color=colors[i], alpha=0.1)

# Set axis labels with larger font size
plt.xlabel('Steps', fontsize=13, fontweight='bold')
plt.ylabel('Probability', fontsize=13, fontweight='bold')
# plt.title('Curriculum Selection Probability', fontsize=16)

# Convert x-axis to k units
ax = plt.gca()
ax.set_xticks([12000, 14000, 16000, 18000, 20000])
ax.set_xticklabels(['12k', '14k', '16k', '18k', '20k'])


ax.set_facecolor('#FAFAFA')
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#333333')
    ax.spines[spine].set_linewidth(1.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Set y-axis limit to 0.7 to give more space for legend
plt.ylim(0.1, 0.7)

# Increase tick label font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid
plt.grid(alpha=0.3)

# Set legend higher - move it closer to the top
plt.legend(loc='upper center', fontsize=8, ncol=4, 
          bbox_to_anchor=(0.5, 1.02))

# Adjust layout
plt.tight_layout()

# Save as 600dpi PDF
plt.savefig('curriculum_selection_probability.pdf', dpi=600, bbox_inches='tight')

# Close figure to free memory
plt.close()

print("Figure saved as curriculum_selection_probability.pdf")