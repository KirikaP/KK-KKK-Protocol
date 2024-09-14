import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def run_experiments(N_values, B_values, num_runs=5000):
    L, K = 3, 3  # Parameters for TreeParityMachine
    all_results = {}

    for N in N_values:
        average_steps = []

        for B in B_values:
            results = train_TPMs(
                L, K, N,
                zero_replace_1=1,
                zero_replace_2=-1,
                num_runs=num_runs,
                rule='anti_hebbian',
                state='anti_parallel',
                B=B
            )
            avg_steps = np.mean(results)
            average_steps.append(avg_steps)
            print(f"N={N}, B={B}: Average Steps={avg_steps}")

        all_results[N] = average_steps

    return B_values, all_results


def save_results_to_csv(B_values, all_results, file_path):
    df = pd.DataFrame(all_results, index=B_values)
    df.index.name = "Bit Package Size (B)"    
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")


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
    plt.xlabel("Bit Package Size (B) [Log Base 2]")
    plt.ylabel("Average Synchronization Steps")
    plt.ylim(y_min_data, y_max_data)
    plt.yticks(y_ticks)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper center')
    plt.tight_layout()    
    plt.savefig("./figures/transparent/t_sync_bit_package.png", transparent=True)
    plt.show()

    save_results_to_csv(B_values, all_results, "./result/bit_package.csv")
