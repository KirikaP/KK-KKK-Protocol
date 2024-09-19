import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_simple_attack(csv_file, bin_width=0.01):
    data = pd.read_csv(csv_file)
    
    N_values = sorted(data['N'].unique())
    
    fig, ax = plt.subplots()

    for N in N_values:
        subset = data[data['N'] == N]
        truncated_ratios = subset[subset['Ratio'] < 1.0]['Ratio']

        success_rate = subset['Success Rate'].iloc[0]

        ax.hist(
            truncated_ratios,
            bins=np.arange(0, 1 + 0.005, bin_width),
            alpha=0.7, label=f'N = {N} ({success_rate:.2%})',
            edgecolor='black', histtype='stepfilled'
        )

    ax.set_xlabel('Ratio(r) (sync_steps / attack_sync_steps)')
    ax.set_ylabel('P(r)')
    ax.grid(True, linestyle='--', alpha=0.3)

    ax_inset = inset_axes(ax, width="75%", height="15%", loc='right')

    for N in N_values:
        subset = data[data['N'] == N]
        ratio_gt_1 = subset[subset['Ratio'] > 1.0]['Ratio']

        if not ratio_gt_1.empty:
            ax_inset.hist(
                ratio_gt_1,
                bins=np.arange(1, ratio_gt_1.max() + bin_width, bin_width),
                alpha=0.7, edgecolor='black', histtype='stepfilled'
            )

    ax_inset.set_xlabel('Ratio > 1.0')
    ax_inset.set_ylabel('P(r)')
    ax_inset.grid(True, linestyle='--', alpha=0.3)

    ax.legend(title="N Values", loc='best')
    plt.tight_layout()

    plt.savefig("./figures/transparent/attack_simple.png", transparent=True)

    plt.show()

if __name__ == "__main__":
    csv_file = "./result/simple_attack.csv"
    plot_simple_attack(csv_file)
