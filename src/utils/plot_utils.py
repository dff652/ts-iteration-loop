
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from pathlib import Path

def generate_ts_thumbnail(data, save_path: str):
    """
    Generate a thumbnail image for time series data with consistent styling.
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data.
        save_path (str): Path to save the image.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine values to plot
        if isinstance(data, pd.DataFrame):
            # If multiple columns, pick the first one or assume single column
            values = data.iloc[:, 0].values
            index = data.index
        else:
            values = data.values
            index = data.index

        fig, ax = plt.subplots(figsize=(10, 2), dpi=100) # Size similar to previous scripts
        
        # Plot data
        ax.plot(index, values, color='black', alpha=1.0, linewidth=0.8)
        
        # Set ticks
        n_ticks = 15
        if len(data) <= n_ticks:
            ax.set_xticks(range(len(data)))
        else:
            tick_positions = np.linspace(0, len(data)-1, n_ticks, dtype=int)
            ax.set_xticks(tick_positions)

        # ax.set_xticklabels([]) # Keep labels visible but small
        # ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Set Y axis range
        y_min, y_max = values.min(), values.max()
        if y_max > y_min:
            y_range = y_max - y_min
            margin = y_range * 0.05
            ax.set_ylim(y_min - margin, y_max + margin)
        
        # Axis formatting
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(axis='both', which='both', direction='in', length=3, width=0.8, pad=2, labelsize=8)
        
        # Beautify
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        # Save
        fig.tight_layout(pad=0.1)
        fig.savefig(save_path, format='jpg', dpi=200, 
                   bbox_inches='tight', pad_inches=0.02,
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to generate plot: {save_path} | Error: {e}")
        return False
