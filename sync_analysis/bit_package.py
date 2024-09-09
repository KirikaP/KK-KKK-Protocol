import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def run_experiments(N_values, B_values, num_runs=5000):
    """
    Run synchronization experiments for different N and B values

    Args:
        N_values (list of int): Different N values (number of input bits per hidden unit)
        B_values (list of int or None): Different bit package sizes
        num_runs (int): Number of synchronization runs per experiment

    Returns:
        tuple: (B_values, all_results), where all_results is a dict with N as keys
               and corresponding average synchronization steps as values
    """
    L, K = 3, 3  # Parameters for TreeParityMachine
    all_results = {}

    for N in N_values:
        average_steps = []

        for B in B_values:
            # Call train_TPMs with B=None for normal sync or a specific value of B for bit package sync
            results = train_TPMs(
                L, K, N,
                zero_replace_1=1,
                zero_replace_2=-1,
                num_runs=num_runs,
                rule='anti_hebbian',
                state='anti_parallel',
                B=B  # Use bit package sync if B is provided, else regular sync
            )
            avg_steps = np.mean(results)
            average_steps.append(avg_steps)
            print(f"N={N}, B={B}: Average Steps={avg_steps}")

        all_results[N] = average_steps

    return B_values, all_results

if __name__ == "__main__":
    B_values = [1, 2, 4, 8, 16, 32, 64, 128]
    N_values = [10, 20, 100]
    num_runs = 5000

    available_markers = ['o', 's', '^', 'd', 'x', '*']
    markers = available_markers * ((len(N_values) // len(available_markers)) + 1)

    B_values, all_results = run_experiments(N_values, B_values, num_runs)

    all_y_values = np.concatenate([all_results[N] for N in N_values])

    y_min_data = min(all_y_values) - 50
    y_max_data = max(all_y_values) + 50
    y_min = 50 * (np.floor(y_min_data / 50))
    y_max = 50 * (np.ceil(y_max_data / 50))
    y_ticks = np.arange(y_min, y_max + 50, 50)

    for i, N in enumerate(N_values):
        plt.plot(
            B_values, 
            all_results[N], 
            marker=markers[i], 
            label=f"N={N}", 
            markerfacecolor='none', 
            linestyle='--'
        )

    plt.xscale('log', base=2)
    plt.title("Synchronization Steps vs. Bit Package Size (B) for Different N")
    plt.xlabel("Bit Package Size (B) [Log Base 2]")
    plt.ylabel("Average Synchronization Steps")

    plt.ylim(y_min_data, y_max_data)
    plt.yticks(y_ticks)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
