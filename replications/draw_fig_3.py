import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.patches import Rectangle
import re

def plot_all_models_by_task(file_path, sheet_name=0, output_dir="model_comparison_plots"):
    """
    Read Excel data and create a single plot showing all models' performance by task.
    Tasks are grouped by difficulty level and displayed with proper formatting.
    Using Seaborn default theme and colors.
    """
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)
    # sns.set_theme(style="whitegrid", palette="husl", font_scale=1.2)
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read Excel file: {file_path}")
        
        # Format task names and create mapping for display
        task_display_names = {}
        for i, task in enumerate(df.iloc[:, 0]):
            if pd.isna(task):
                continue
            task_display_names[i] = task
        
        # Define task groups by difficulty
        task_groups = {
            "Easy": [],
            "Middle": [],
            "Hard": [],
            "None": []
        }
        
        # Map tasks to their groups
        task_to_group = {
            "Collect Wood": "Easy",
            "Collect Sapling": "Easy",
            "Eat Plant": "Easy",
            "Make Wood Pickaxe": "Easy",
            "Make Wood Sword": "Easy",
            "Place Plant": "Middle",
            "Wake Up": "Easy",
            
            "Collect Stone": "Middle",
            "Collect Iron": "Hard",
            "Collect Coal": "Middle",
            "Make Stone Pickaxe": "Middle",
            "Make Stone Sword": "Middle",
            "Place Stone": "Middle",
            "Place Table": "Middle",
            "Eat Cow": "Middle",
            
            "Collect Diamond": "Hard",
            "Defeat Skeleton": "Hard",
            "Collect Drink": "Easy",
            "Make Iron Pickaxe": "Hard",
            "Make Iron Sword": "Hard",
            "Place Furnace": "Hard",
            "Defeat Zombie": "Hard",
            "Final Score": "None",
        }

        
        # Populate task groups
        for i, task in enumerate(df.iloc[:, 0]):
            if pd.isna(task):
                continue
                
            task_str = str(task)
            if task_str in task_to_group:
                group = task_to_group[task_str]
                task_groups[group].append(i)
        
        # Extract model names from column headers, excluding the first column (task names)
        model_names = df.columns[1:6].tolist()
        
        # Create a DataFrame for plotting with a new column for position
        plot_data = []
        
        # Calculate positions with gaps between groups
        group_names = ["Easy", "Middle", "Hard", "None"]  # Ensure correct order
        position = 0
        position_map = {}  # Maps task index to x-position
        
        # First pass: assign positions
        for group_name in group_names:
            task_indices = task_groups[group_name]
            for i, task_idx in enumerate(task_indices):
                position_map[task_idx] = position
                position += 1
            # Add gap after group
            position += 0.5  # Larger gap between groups
        
        # Create plotting data
        for i, task in enumerate(df.iloc[:, 0]):
            if pd.isna(task):
                continue
                
            for j, model in enumerate(model_names):
                score = df.iloc[i, j+1] * 100
                # Handle zeros or very small numbers for logarithmic scale
                if score <= 0:
                    print(f"Warning: Value {score} for task '{task}', model '{model}' is ≤ 0. Setting to 0.01 for log scale.")
                    score = 0.01
                elif score < 0.01:
                    score = 0.01  # Set minimum value for log scale
                
                # Use position map or default to index if not found
                pos = position_map.get(i, i)
                
                plot_data.append({
                    'Task': task_display_names.get(i, str(task)),  # Use formatted task name
                    'RawTask': str(task),  # Keep raw task name for reference
                    'Model': model,
                    'Score': score,
                    'Position': pos,
                    'Group': task_to_group.get(str(task), "Other")  # Assign group
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Set Seaborn style and color palette
        # sns.set_theme()  # Use default Seaborn theme
        
        # Create the figure
        plt.figure(figsize=(18, 5))
        
        # Create custom plot with manual positioning
        ax = plt.subplot(111)
        plt.subplots_adjust(top=0.75)  
        plt.tight_layout(rect=[0, 0, 1, 0.9])  
        
        # Calculate width for each bar within a position
        num_models = len(model_names)
        bar_width = 0.8 / num_models
        
        # Get Seaborn default color palette
        colors = sns.color_palette(n_colors=len(model_names))
        
        # Plot bars manually for precise control
        for i, model in enumerate(model_names):
            model_data = plot_df[plot_df['Model'] == model]
            
            # Sort by position
            model_data = model_data.sort_values('Position')
            
            # Calculate bar positions
            positions = model_data['Position']
            x_positions = positions + (i - num_models/2 + 0.5) * bar_width
            
            # Plot bars with Seaborn color
            bars = ax.bar(x_positions, model_data['Score'], width=bar_width, 
                   label=model, alpha=0.85, color=colors[i])
            
            # Add value labels on the bars
            # for x, y in zip(x_positions, model_data['Score']):
            #     ax.text(x, y * 1.05, f'{y:.2f}', ha='center', va='bottom', 
            #            fontsize=8, rotation=90)
        
        # Set log scale
        ax.set_yscale('log')
        ax.set_ylim(0.01, 100)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'],fontsize=16)
        
        # Get subtle background colors from Seaborn palette
        palette = sns.color_palette("pastel", 4)
        group_colors = {
            "Easy": palette[0],
            "Middle": palette[1],
            "Hard": palette[2],
            "None": palette[3]
        }
        
        # Add group separators, backgrounds, and labels
        prev_end = -0.5
        group_positions = []
        
        # Calculate group boundaries and add visual elements
        for group_name in group_names:
            task_indices = task_groups[group_name]
            if not task_indices:
                continue
                
            # Get min and max positions for this group
            positions = [position_map.get(idx, idx) for idx in task_indices]
            if not positions:
                continue
                
            start_pos = min(positions) - 0.5
            end_pos = max(positions) + 0.5
            mid_pos = (start_pos + end_pos) / 2
            
            # Add vertical separator line after each group (except the last)
            if prev_end >= 0:
                ax.axvline(x=prev_end + (start_pos - prev_end)/2, color='#DDDDDD', linestyle='-', alpha=0.7)
            
            # Add background for group with Seaborn colors
            color = group_colors.get(group_name)
            rect = Rectangle((start_pos, 0.008), end_pos - start_pos, 100/0.008, 
                            fill=True, alpha=0.15, color=color)
            ax.add_patch(rect)
            
            # Add group label
            group_positions.append((mid_pos, group_name))
            prev_end = end_pos
        
        # Set x-ticks at task positions with rotated labels
        task_positions = []
        task_labels = []
        for task_idx, position in position_map.items():
            task_positions.append(position)
            task_labels.append(task_display_names.get(task_idx, str(df.iloc[task_idx, 0])))
            
        ax.set_xticks(task_positions)
        ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=18)
        
        # Add a second x-axis for group labels
        # ax2 = ax.twiny()
        # ax2.set_xlim(ax.get_xlim())
        # print(group_positions)
        # ax2.set_xticks([pos for pos, _ in group_positions])
        # ax2.set_xticklabels([name for _, name in group_positions], fontsize=12, fontweight='bold')
        # ax2.tick_params(axis='x', which='major', pad=15)
        
        # # Make the top ticks invisible
        # ax2.tick_params(axis='x', which='both', bottom=False, top=False)
        
        # Add title and labels with improved styling
        # plt.title('Model Performance Comparison by Task Difficulty', fontsize=16, pad=30)

        # ax.set_xlabel('Tasks', fontsize=12, labelpad=10)
        all_positions = list(position_map.values())
        min_pos = min(all_positions) - 0.5
        max_pos = max(all_positions) + 0.5
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylabel('Score (log scale)', fontsize=20)
        
        # Add legend with improved styling
        legend = ax.legend(
            title='Models',
            loc='upper center',     
            bbox_to_anchor=(0.5, 1.3),  
            ncol=len(model_names),      
            frameon=True,             
            title_fontsize=20,        
            fontsize=18,                 
        )
                
        # Save the figure
        plt.savefig(f"task_difficulty_comparison.pdf", dpi=1200, bbox_inches='tight')
        print(f"Saved plot with Seaborn default theme")
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the script
if __name__ == "__main__":
    file_path = "fig4_data_export_2025-06-17T14_20_32.412+8_00.xlsx"
    
    if os.path.exists(file_path):
        plot_all_models_by_task(file_path)
    else:
        print(f"File not found: {file_path}")