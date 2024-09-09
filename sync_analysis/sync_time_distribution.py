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
    """
    Filter the step counts for synchronization up to max_t_sync for different N values

    Args:
        L (int): Weight limit range [-L, L]
        K (int): Number of hidden units
        N_values (list): List of different N values (number of input bits)
        num_runs (int): Number of runs for each N value
        max_t_sync (int): Maximum synchronization steps to consider
        zero_replace_1 (int): Value to replace 0 in tpm1's sigma
        zero_replace_2 (int): Value to replace 0 in tpm2's sigma
        rule (str): Learning rule for the sync process
        state (str): Synchronization state

    Returns:
        List of filtered step counts for each N value
    """
    N_step_counts = []
    for N in N_values:
        print(f"Running N = {N}")
        # Train TPMs for the current N value
        step_counts = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        # Filter out steps that are greater than max_t_sync
        filtered_step_counts = [count for count in step_counts if count <= max_t_sync]
        N_step_counts.append(filtered_step_counts)

    return N_step_counts

def plot_results(N_values, N_step_counts, L, K, bin_width=30, colors=None):
    """
    Plot histograms for each N value showing the distribution of t_sync

    Args:
        N_values (list): List of different N values (number of input bits)
        N_step_counts (list): List of step counts for each N value
        L, K (int): Weight limit range [-L, L] and number of hidden units
        bin_width (int): Width of each histogram bin
        colors (list): List of colors for each N value histogram
    """
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
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}')
    plt.legend(loc='upper right')
    plt.xlim(0, 3000)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Define parameters
    L, K, N_values = 3, 3, [10, 100, 1000]
    num_runs = 5000  # Number of runs per N value

    # Run the simulation and filter step counts
    N_step_counts = simulate(L, K, N_values, num_runs=num_runs)

    # Plot the histograms of filtered t_sync values
    plot_results(N_values, N_step_counts, L, K)
