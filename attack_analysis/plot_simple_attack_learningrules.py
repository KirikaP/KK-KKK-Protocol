import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

def plot_simple_attack(csv_file, bin_width=0.01):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist.")
        return
    
    data = pd.read_csv(csv_file)
    
    rules = data['Rule'].unique()
    
    fig, ax = plt.subplots()

    for rule in rules:
        subset = data[data['Rule'] == rule]
        truncated_ratios = subset[subset['Ratio'] < 1.0]['Ratio']

        success_rate = subset['Success Rate'].iloc[0]

        ax.hist(
            truncated_ratios,
            bins=np.arange(0, 1 + 0.005, bin_width),
            alpha=0.7, label=f'{rule.capitalize()} ({success_rate:.2%})',
            edgecolor='black', histtype='stepfilled'
        )

    ax.set_xlabel('Ratio (r) (sync_steps / attack_sync_steps)')
    ax.set_ylabel('P(r)')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Insert small plot to show distribution of Ratio > 1.0
    ax_inset = inset_axes(ax, width="75%", height="15%", loc='right')

    for rule in rules:
        subset = data[data['Rule'] == rule]
        ratio_gt_1 = subset[subset['Ratio'] > 1.0]['Ratio']

        if not ratio_gt_1.empty:
            ax_inset.hist(
                ratio_gt_1,
                bins=np.arange(1, ratio_gt_1.max() + bin_width, bin_width),
                alpha=0.7, edgecolor='black', histtype='stepfilled', label=f'{rule.capitalize()}'
            )

    ax_inset.set_xlabel('Ratio > 1.0')
    ax_inset.set_ylabel('P(r)')
    ax_inset.grid(True, linestyle='--', alpha=0.3)

    ax.legend(title="Learning Rules", loc='best')
    plt.tight_layout()

    # Ensure the directory exists
    output_file = "./figures/transparent/attack_simple_various_rules.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.savefig(output_file, transparent=True)

    plt.show()

if __name__ == "__main__":
    csv_file = "./result/simple_attack_various_rules.csv"  # CSV file with different rules
    plot_simple_attack(csv_file)
