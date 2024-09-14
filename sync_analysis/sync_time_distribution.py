import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def simulate(
    L, K, N_values, num_runs=5000, max_t_sync=3000, zero_replace_1=1, zero_replace_2=-1,
    rule='anti_hebbian', state='anti_parallel'
):
    N_step_counts = []
    for N in N_values:
        print(f"Running N = {N}")
        step_counts = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        # Filter out steps that are greater than max_t_sync
        filtered_step_counts = [count for count in step_counts if count <= max_t_sync]
        N_step_counts.append(filtered_step_counts)

    return N_step_counts

def plot_results(N_values, N_step_counts, L, K, bin_width=30, colors=None):
    if colors is None:
        colors = ['green', 'orange', 'black']  # Default colors

    for N, color, step_counts in zip(N_values, colors, N_step_counts):
        bins = np.arange(0, 3000 + bin_width, bin_width)
        plt.hist(
            step_counts,
            bins=bins,
            color=color,
            label=f'N = {N}',
            histtype='stepfilled' if N in [10, 100] else 'step',
            alpha=0.5 if N in [10, 100] else 1
        )

    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.legend(loc='upper right')
    plt.xlim(0, 3000)
    plt.grid(True)
    plt.savefig('./figures/transparent/t_sync_distribution.png', transparent=True)
    plt.show()


if __name__ == "__main__":
    L, K, N_values = 3, 3, [10, 100, 1000]
    num_runs = 5000

    N_step_counts = simulate(L, K, N_values, num_runs=num_runs)

    plot_results(N_values, N_step_counts, L, K)
