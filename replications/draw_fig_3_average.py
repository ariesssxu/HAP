import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_group_logavg_by_task(file_path, sheet_name=0, output_dir="model_comparison_plots"):
    """
    Read Excel data, for each group, for each model:
    Take log10(score) (min capped), then average across tasks, plot group-average log10(score).
    """
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)
    try:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read Excel file: {file_path}")

        # Define mapping from task to group
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
            "Final Score": "Final Score",
        }

        model_names = df.columns[1:6].tolist()
        group_names = ["Easy", "Middle", "Hard", "Final Score"]

        # 收集每组每模型的log均值
        plot_data = []
        group_indices = {g: [] for g in group_names}
        for i, task in enumerate(df.iloc[:, 0]):
            if pd.isna(task):
                continue
            group = task_to_group.get(str(task))
            if group in group_names:
                group_indices[group].append(i)

        for group in group_names:
            indices = group_indices[group]
            if not indices:
                continue
            for model in model_names:
                vals = (df.loc[indices, model].values) * 100
                # 防止0或负数报错，最小为0.01
                vals = np.clip(vals, 0.01, None)
                logvals = np.log10(vals)
                avg_logval = np.mean(logvals)
                plot_data.append({
                    "Group": group,
                    "Model": model,
                    "AvgLogScore": avg_logval
                })
        plot_df = pd.DataFrame(plot_data)

        # 画图
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        bar_width = 0.75 / len(model_names)
        group_pos = np.arange(len(group_names))
        colors = sns.color_palette(n_colors=len(model_names))

        for i, model in enumerate(model_names):
            scores = []
            for group in group_names:
                val = plot_df[(plot_df["Group"] == group) & (plot_df["Model"] == model)]["AvgLogScore"]
                scores.append(val.iloc[0] if len(val) > 0 else np.nan)
            x_offset = group_pos + (i - len(model_names)/2 + 0.5) * bar_width
            bars = ax.bar(
                x_offset, 
                np.array(scores) - (-2),    # bar高度为top-bottom，确保起点在-2
                width=bar_width, 
                label=model, 
                color=colors[i], 
                alpha=0.88,
                bottom=-2                   # bar底部为-2
            )
        ax.set_ylim(-2, 2)

        # y轴是log10(score)
        ax.set_ylim(-2, 2)  # 对应0.01~100
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['log10(0.01)', 'log10(0.1)', 'log10(1)', 'log10(10)', 'log10(100)'], fontsize=18)
        ax.set_xticks(group_pos)
        ax.set_xticklabels(group_names, fontsize=18)
        ax.set_ylabel('Average log10(Score)', fontsize=20)
        ax.set_xlabel('Task Difficulty', fontsize=20)
        ax.legend(
            title='Models',
            loc='upper center',     
            bbox_to_anchor=(0.5, 1.24),  
            ncol=len(model_names),      
            frameon=True,             
            title_fontsize=18,        
            fontsize=16,      
            columnspacing=0.8,           # 列间距更紧凑
            handletextpad=0.4            # 图例标记与文本的间距再小一点           
        )
        # plt.title('Mean log10(Score) per Model, per Task Group', fontsize=18, pad=20)
        plt.savefig("task_difficulty_group_logavg.pdf", dpi=1200, bbox_inches='tight')
        print(f"Saved log-average plot as PDF")

    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        import traceback
        traceback.print_exc()

# Run
if __name__ == "__main__":
    file_path = "fig4_data_export_2025-06-17T14_20_32.412+8_00.xlsx"
    if os.path.exists(file_path):
        plot_group_logavg_by_task(file_path)
    else:
        print(f"File not found: {file_path}")
