import matplotlib.pyplot as plt
import numpy as np

# Steps
steps = range(10)

with open('human_study_results_hap.xlsx', 'rb') as f:
    # Read the Excel file
    import pandas as pd
    df = pd.read_excel(f, sheet_name=None).get('Sheet1')

# Manually enter the data
results = []

for line in df.values:
    # Skip the first row (header)
    if line[0] == 'Step':
        continue
    # Append the data to the respective lists
    results.append([float(x) for x in line[1:11]])

# Convert to numpy arrays
without_tutorial, with_expert_tutorial, with_auto_tutorial = results[0:10], results[10:20], results[20:30]
without_tutorial = np.array(without_tutorial)
with_expert_tutorial = np.array(with_expert_tutorial)
with_auto_tutorial = np.array(with_auto_tutorial)

# Calculate means and standard errors
without_tutorial_mean = np.mean(without_tutorial, axis=0)
without_tutorial_se = np.std(without_tutorial, axis=0) / np.sqrt(len(without_tutorial))

with_expert_tutorial_mean = np.mean(with_expert_tutorial, axis=0)
with_expert_tutorial_se = np.std(with_expert_tutorial, axis=0) / np.sqrt(len(with_expert_tutorial))

with_auto_tutorial_mean = np.mean(with_auto_tutorial, axis=0)
with_auto_tutorial_se = np.std(with_auto_tutorial, axis=0) / np.sqrt(len(with_auto_tutorial))

# Calculate min and max for each step (for shaded area)
without_tutorial_min = np.min(without_tutorial, axis=0)
without_tutorial_max = np.max(without_tutorial, axis=0)

with_expert_tutorial_min = np.min(with_expert_tutorial, axis=0)
with_expert_tutorial_max = np.max(with_expert_tutorial, axis=0)

with_auto_tutorial_min = np.min(with_auto_tutorial, axis=0)
with_auto_tutorial_max = np.max(with_auto_tutorial, axis=0)

# Create a figure
plt.figure(figsize=(7, 5))

# Define primary colors for each condition
red_primary = 'red'
yellow_primary = '#DAA520'  # Goldenrod
green_primary = '#228B22'  # Forest Green

# Create color gradients from light to dark
red_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(steps)))
yellow_colors = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(steps)))
green_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(steps)))

# Plot shaded regions for data range (min-max)
plt.fill_between(steps, 
                without_tutorial_min, 
                without_tutorial_max, 
                color=red_primary, alpha=0.1, label='_nolegend_')

plt.fill_between(steps, 
                with_expert_tutorial_min, 
                with_expert_tutorial_max, 
                color=yellow_primary, alpha=0.1, label='_nolegend_')

plt.fill_between(steps, 
                with_auto_tutorial_min, 
                with_auto_tutorial_max, 
                color=green_primary, alpha=0.1, label='_nolegend_')

# Plot shaded regions for standard errors
plt.fill_between(steps, 
                without_tutorial_mean - without_tutorial_se, 
                without_tutorial_mean + without_tutorial_se, 
                color=red_primary, alpha=0.3, label='_nolegend_')

plt.fill_between(steps, 
                with_expert_tutorial_mean - with_expert_tutorial_se, 
                with_expert_tutorial_mean + with_expert_tutorial_se, 
                color=yellow_primary, alpha=0.3, label='_nolegend_')

plt.fill_between(steps, 
                with_auto_tutorial_mean - with_auto_tutorial_se, 
                with_auto_tutorial_mean + with_auto_tutorial_se, 
                color=green_primary, alpha=0.3, label='_nolegend_')

# Plot lines connecting the means
plt.plot(steps, without_tutorial_mean, '-', color=red_primary, alpha=0.7, linewidth=2)
plt.plot(steps, with_expert_tutorial_mean, '-', color=yellow_primary, alpha=0.7, linewidth=2)
plt.plot(steps, with_auto_tutorial_mean, '-', color=green_primary, alpha=0.7, linewidth=2)

# Plot each data point with gradient colors
for i in range(len(steps)):
    # Without tutorial (red)
    plt.plot(steps[i], without_tutorial_mean[i], 
            marker='o', color=red_colors[i], markersize=8,
            markeredgecolor='black', markeredgewidth=0.5)
    
    # With expert tutorial (yellow)
    plt.plot(steps[i], with_expert_tutorial_mean[i], 
            marker='s', color=yellow_colors[i], markersize=8,
            markeredgecolor='black', markeredgewidth=0.5)
    
    # With auto tutorial (blue)
    plt.plot(steps[i], with_auto_tutorial_mean[i], 
            marker='^', color=green_colors[i], markersize=8,
            markeredgecolor='black', markeredgewidth=0.5)

# Custom legend for the three conditions
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Create legend elements
legend_elements = [
    # Lines for mean values
    Line2D([0], [0], marker='o', color=red_primary, alpha=0.8, markersize=8, 
          markeredgecolor='black', markeredgewidth=0.5, label='Without Tutorial'),
    Line2D([0], [0], marker='s', color=yellow_primary, alpha=0.8, markersize=8, 
          markeredgecolor='black', markeredgewidth=0.5, label='With Expert Tutorial'),
    Line2D([0], [0], marker='^', color=green_primary, alpha=0.8, markersize=8, 
          markeredgecolor='black', markeredgewidth=0.5, label='With Auto Tutorial'),
]

# Add explanation patches
dark_shade = Patch(facecolor='gray', alpha=0.3, label='Standard Error (SE)')
light_shade = Patch(facecolor='gray', alpha=0.1, label='Min-Max Range')
legend_elements.extend([dark_shade, light_shade])

# Add text explanation for color gradient
plt.figtext(0.42, 0.18, "Color darkness increases with step number", 
           ha="left", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

# Add labels and legend
plt.xlabel('Step', fontsize=22)
# set x-axis ticks font size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Score', fontsize=22)
# plt.title('Human Study Results', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(handles=legend_elements, fontsize=14, loc='upper left')
plt.xticks(steps)

# Set y-axis limits
plt.ylim(0, 5)

# Show the plot
plt.tight_layout()
plt.savefig('human_study_results.pdf', dpi=300, bbox_inches='tight')
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# labels = ['ambiguous button', 'key', 'door']
# data = np.array([
#     [6, 8, 7],   # without tutorial
#     [8, 10, 9],  # with expert tutorial
#     [7, 10, 10], # with auto tutorial
# ])

# x = np.arange(len(labels))    # [0,1,2]
# width = 0.22                  # 每组三个柱子的宽度

# red_primary = 'red'
# yellow_primary = '#DAA520'
# green_primary = '#228B22'

# colors = [red_primary, yellow_primary, green_primary]
# bar_labels = ['Without Tutorial', 'With Expert Tutorial', 'With Auto Tutorial']

# plt.figure(figsize=(7, 5))

# for i, (color, blabel) in enumerate(zip(colors, bar_labels)):
#     plt.bar(x + width * (i - 1), data[i], width=width, color=color,
#             edgecolor='black', linewidth=1.0, alpha=0.7, label=blabel)

# plt.xlabel('Object', fontsize=14)
# plt.ylabel('Count', fontsize=14)
# plt.xticks(x, labels, fontsize=12)
# plt.ylim(0, max(data.flatten()) + 2)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.legend(fontsize=12)
# plt.tight_layout()

# plt.savefig('bar_human_study.pdf', dpi=300, bbox_inches='tight')
# plt.show()
