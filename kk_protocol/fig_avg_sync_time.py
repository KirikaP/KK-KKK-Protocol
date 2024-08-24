import matplotlib.pyplot as plt
import numpy as np
from scripts.party_machine import PM, train_PMs


def simulate(L, K, N_values):
    results = []
    for N in N_values:
        pm1 = PM(L, N, K, zero_replace=1)
        pm2 = PM(L, N, K, zero_replace=-1)

        sync_steps = train_PMs(pm1, pm2)
        mean_steps = np.mean(sync_steps)
        results.append(mean_steps)

    return results


if __name__ == "__main__":
    L, K, N_values = 3, 3, [10, 17, 31, 100, 316, 1000]
    avg_sync_times = simulate(L, K, N_values)

    for N, y in zip(N_values, avg_sync_times):
        x = np.log10(N)
        plt.scatter(x, y, edgecolors='black', facecolors='none', label=f'{avg_sync_times}')
        plt.text(x, y, f'{y:.0f}', ha='left', va='bottom')

    plt.xlim([np.log10(min(N_values)) - 0.2, np.log10(max(N_values)) + 0.2])
    plt.xlabel('log_10(N)')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time vs N (L = {L}, K = {K})')
    plt.grid(True)
    plt.show()
