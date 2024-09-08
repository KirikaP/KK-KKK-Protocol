from scripts.parity_machine import TreeParityMachine as TPM
from scripts.kk_protocol import sync_with_bit_packages, train_TPMs_with_bit_packages
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_experiments(N_values, B_values, num_runs=5000):
    L, K = 3, 3  # Example parameters for TPM
    all_results = {}

    for N in N_values:
        average_steps = []

        for B in B_values:
            sender = TPM(L, N, K, 1)
            receiver = TPM(L, N, K, -1)

            results = train_TPMs_with_bit_packages(sender, receiver, B, num_runs, rule='anti_hebbian', state='anti_parallel')
            avg_steps = np.mean(results)
            average_steps.append(avg_steps)
            print(f"N={N}, B={B}: {avg_steps}")
        
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
    y_min = 50 * (np.floor(y_min_data / 50))  # Round down to nearest 50
    y_max = 50 * (np.ceil(y_max_data / 50))   # Round up to nearest 50
    y_ticks = np.arange(y_min, y_max + 50, 50)

    for i, N in enumerate(N_values):
        plt.plot(B_values, all_results[N], marker=markers[i], label=f"N={N}", markerfacecolor='none', linestyle='--')

    plt.xscale('log', base=2)  # Set x-axis to log scale with base 2 for B values
    plt.title("Synchronization Steps vs. Bit Package Size (B) for Different N")
    plt.xlabel("Bit Package Size (B) [Log Base 2]")
    plt.ylabel("Average Synchronization Steps")

    plt.ylim(y_min_data, y_max_data)
    plt.yticks(y_ticks)

    plt.grid(True, linestyle='--', alpha=0.3)  # Optional: grid lines
    plt.legend()
    plt.show()
