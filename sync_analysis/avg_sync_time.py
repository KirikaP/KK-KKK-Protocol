import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def simulate(
    L, K, N_values, num_runs=5000, zero_replace_1=1, zero_replace_2=-1,
    rule='anti_hebbian', state='anti_parallel'
):
    results = []
    for N in N_values:
        print(f"Running N = {N}")
        sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        mean_steps = np.mean(sync_steps)
        results.append(mean_steps)

    return results

def plot_results(N_values, avg_sync_times, L, K):
    for N, y in zip(N_values, avg_sync_times):
        x = 1 / N
        plt.scatter(x, y, color='black', marker='o')

    y_max = max(avg_sync_times)
    plt.ylim([0, y_max + 0.1 * y_max])
    plt.xlim([1 / max(N_values) - 0.01, 1 / min(N_values) + 0.01])
    plt.xlabel('1/N')
    plt.ylabel('Average Synchronization Steps')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.savefig('./figures/transparent/t_sync_with_N.png', transparent=True)
    plt.show()

def save_results_to_csv(N_values, avg_sync_times, file_path):
    df = pd.DataFrame({
        'N': N_values,
        'Average Synchronization Steps': avg_sync_times
    })
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    L, K, N_values = 3, 3, [1000, 100, 50, 20, 10]
    num_runs = 5000

    avg_sync_times = simulate(L, K, N_values, num_runs)

    plot_results(N_values, avg_sync_times, L, K)
    save_results_to_csv(N_values, avg_sync_times, './results/avg_sync_time.csv')
