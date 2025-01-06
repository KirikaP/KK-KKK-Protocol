import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs

# parameters
L = 3  # Weight range [-L, L]
K = 3  # Number of hidden units
N_values = [10, 100, 1000]  # Different values of N for the simulation
num_runs = 5000  # Number of simulations to run
max_t_sync = 3000  # Maximum synchronization steps to consider
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
rule = 'random_walk'  # Learning rule
state = 'parallel'  # Synchronization state
bin_width = 30  # Width of histogram bins
colors = ['green', 'orange', 'black']  # Colors for histograms
figure_file = './figures/transparent/t_sync_distribution_rw.png'  # File to save the figure

def simulate(
    L, K, N_values, num_runs=5000, max_t_sync=3000, zero_replace_1=-1, zero_replace_2=-1,
    rule='random_walk', state='parallel'
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
        bins = np.arange(0, max_t_sync + bin_width, bin_width)
        plt.hist(
            step_counts,
            bins=bins,
            color=color,
            label=f'N = {N}',
            histtype='stepfilled' if N in [10, 100] else 'step',
            alpha=0.5 if N in [10, 100] else 1
        )

    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.xlim(0, max_t_sync)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    plt.savefig(figure_file, transparent=True)
    plt.show()

if __name__ == "__main__":
    N_step_counts = simulate(L, K, N_values, num_runs=num_runs, max_t_sync=max_t_sync)

    plot_results(N_values, N_step_counts, L, K, bin_width=bin_width, colors=colors)
