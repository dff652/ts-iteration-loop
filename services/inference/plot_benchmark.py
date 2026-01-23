import matplotlib.pyplot as plt
import numpy as np
import os

# Data
methods = ['Qwen3-VL-8B-Instruct', 'ChatTS-14B', 'timer-base-84m', 'Adtk_Hbos']
metrics = {
    'Average Time (s)': [61.5, 28.85, 81.26, 128],
    'Average Inference Time (s)': [60.6, 10.59, 61.95, 91],
    'Inference Ratio': [0.985365854, 0.367071057, 0.762367709, 0.7109375],
    'Max GPU Memory (GB)': [20.05, 41.8, 2, 0],
    'Average Score': [21.2, 33.6, 67.2, 67]
}

baseline = 'Adtk_Hbos'
output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

# Plotting settings
# Seaborn muted palette inspired colors: Blue for normal, Orange for baseline
colors = ['#4c72b0' if m != baseline else '#dd8452' for m in methods] 

# --- Combined Plot for Time Metrics ---
def plot_combined_time(methods, metrics, output_dir):
    time_keys = ['Average Time (s)', 'Average Inference Time (s)']
    
    x = np.arange(len(methods))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for the two distinct metrics to differentiate them
    # We can't use the per-algorithm color scheme easily for groups unless we use patterns or shades.
    # Approach: Use distinct colors for Total Time vs Inference Time, and maybe highlight baseline via text or axis label?
    # OR: Keep the per-algorithm color scheme but put them side-by-side? 
    # Usually "grouped bar" means comparing Metric A vs Metric B for each Algorithm.
    # So bars should be colored by Metric.
    
    metric_colors = ['#4c72b0', '#55a868'] # Blue and Green
    
    rects1 = ax.bar(x - width/2, metrics[time_keys[0]], width, label='Average Total Time', color=metric_colors[0], alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, metrics[time_keys[1]], width, label='Average Inference Time', color=metric_colors[1], alpha=0.9, edgecolor='black')

    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Benchmark Analysis: Time Consumption', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_time_analysis.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

plot_combined_time(methods, metrics, output_dir)


# --- Individual Plots for Others ---
# Filter out the time metrics we just combined
remaining_metrics = {k: v for k, v in metrics.items() if k not in ['Average Time (s)', 'Average Inference Time (s)']}

for metric_name, values in remaining_metrics.items():
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title(f'Benchmark Analysis: {metric_name}', fontsize=14, pad=20)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set specific Y-axis limits based on metric
    if metric_name == 'Average Score':
        plt.ylim(0, 100)
    elif metric_name == 'Max GPU Memory (GB)':
        plt.ylim(0, 48)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        # Format label based on value size
        if metric_name == 'Inference Ratio':
             label = f'{height:.4f}'
        elif height == 0:
             label = '0'
        elif height < 1:
             label = f'{height:.3f}'
        else:
             label = f'{height:.2f}'
        
        plt.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save
    # Clean filename
    safe_name = metric_name.split('(')[0].strip().lower().replace(' ', '_')
    save_path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

print(f"All plots completed in {output_dir}")
