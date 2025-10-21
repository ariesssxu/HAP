import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

x = [10000, 25000, 50000, 100000, 150000, 200000]
x_labels = ['10k', '25k', '50k', '100k', '150k', '200k']
groups = ['simple', 'middle', 'hard', 'exrhard']
colors = ['#CE4C4A', '#5280C1', '#46945C', '#E5B136']
labels = ['Simple', 'Middle', 'Hard', 'Extreme Hard']
models = ["hap", "dqn", "sac", "a2c"]
model_titles = ["HAP (ours)", "DQN (exp_fac=0.5)", "SAC", "A2C"]

def extract_all_data(data, group):
    """Extract all run data for box plots and means for line plots"""
    df_group = data[data['Group'] == group]
    run_cols = [c for c in df_group.columns if c.startswith("Run")]
    
    # Get all data points for box plots
    all_data = []
    means = []
    
    for idx in range(len(df_group)):
        row_data = df_group.iloc[idx][run_cols].values
        # Remove NaN values and convert to numeric
        clean_data = pd.to_numeric(row_data, errors='coerce')
        clean_data = clean_data[~np.isnan(clean_data)]
        # Ensure data is in [0, 1] range
        clean_data = np.clip(clean_data, 0, 1)
        all_data.append(clean_data)
        means.append(np.nanmean(clean_data))
    
    return all_data, np.array(means)

# Create narrower figure
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
# fig.suptitle('Tasks', fontsize=16, fontweight='bold', y=0.98)

for idx, (model, title) in enumerate(zip(models, model_titles)):
    ax = axes[idx]
    
    # Read data
    try:
        data = pd.read_csv(f'{model}_data_export.csv')
        print(f"Successfully loaded {model}_data_export.csv")
    except FileNotFoundError:
        print(f"File {model}_data_export.csv not found")
        continue
    
    # Extract data for all groups
    group_data = []
    group_means = []
    
    for group in groups:
        all_data, means = extract_all_data(data, group)
        group_data.append(all_data)
        group_means.append(means)
    
    # Plot box plots and line plots for each group
    box_width = 4000
    positions_base = np.array(x)
    
    for i, (all_data, means, color, label) in enumerate(zip(group_data, group_means, colors, labels)):
        # Reduced offset for more compact layout
        offset = (i - 1.5) * 4000
        positions = positions_base + offset
        
        # Create box plot with shorter whiskers
        if len(all_data) > 0 and all(len(data_point) > 0 for data_point in all_data):
            bp = ax.boxplot(all_data, positions=positions, widths=box_width,
                           patch_artist=True, showfliers=False,
                           whis=0.75)
            
            # Style box plots
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
                patch.set_edgecolor(color)
                patch.set_linewidth(1)
            
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color(color)
                    item.set_linewidth(1.5)
        
        # Plot line connecting means
        ax.plot(positions, means, color=color, label=label, linewidth=3, 
               marker='o', markersize=7, markerfacecolor=color, 
               markeredgecolor='white', markeredgewidth=2, alpha=0.9, zorder=10)
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Set axis style
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Set tick style
    ax.tick_params(axis='x', colors='#333333', labelsize=8, width=1.5)  # Further reduced font
    ax.tick_params(axis='y', colors='#333333', labelsize=11, width=1.5)
    
    # Set labels and ticks with rotation to prevent overlap
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10, 
                      rotation=45, ha='right')  # Rotated labels
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Set title
    ax.set_xlabel(title, fontsize=12, fontweight='bold', labelpad=8)  # Increased padding
    
    # Only show y-axis labels on the first subplot
    if idx == 0:
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                          fontsize=11)
    else:
        ax.set_yticklabels([])
    
    # Set background color
    ax.set_facecolor('#FAFAFA')

# Add legend
handles, labels_legend = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                   ncol=4, frameon=False, fontsize=12, columnspacing=1.2)

# Set legend marker size
for handle in legend.legend_handles:
    handle.set_markersize(8)
    handle.set_linewidth(3)

# Adjust layout for rotated labels - need more bottom space
plt.tight_layout()
plt.subplots_adjust(top=0.82, bottom=0.20, left=0.06, right=0.96, wspace=0.05)  # More bottom space

# Save as 600dpi PDF without popup window
plt.savefig('combined_box_line_plots_narrow_rotated.pdf', dpi=600, bbox_inches='tight', 
           facecolor='white', edgecolor='none')

plt.close()

print("Narrow chart with rotated labels saved as combined_box_line_plots_narrow_rotated.pdf (600 DPI)")