import matplotlib.pyplot as plt
import numpy as np
from scripts.parity_machine import TreeParityMachine as TPM, train_TPMs


def simulate(L, K, N_values):
    results = []
    for N in N_values:
        pm1 = TPM(L, N, K, zero_replace=1)
        pm2 = TPM(L, N, K, zero_replace=-1)

        sync_steps = train_TPMs(pm1, pm2, rule='anti_hebbian')
        mean_steps = np.mean(sync_steps)
        results.append(mean_steps)

    return results

if __name__ == "__main__":
    L, K, N_values = 3, 3, [1000, 100, 50, 20, 10]
    # L, K, N_values = 3, 3, [1000, 100, 10]
    avg_sync_times = simulate(L, K, N_values)

    offsets = {
        1000: (-0.005, -140),
        100: (-0.005, 10),
        50: (0.001, -15),
        20: (0.001, -15),
        10: (-0.005, -140)
    }

    for N, y in zip(N_values, avg_sync_times):
        x = 1 / N
        plt.scatter(x, y, color='black', marker='o')
        # dx, dy = offsets[N]
        # plt.text(x + dx, y + dy, f'N={N}\n{y:.2f}', ha='left', va='bottom')

    y_max = max(avg_sync_times)
    plt.ylim([0, y_max + 0.1 * y_max])
    plt.xlim([1 / max(N_values) - 0.01, 1 / min(N_values) + 0.01])
    plt.xlabel('1/N')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time vs 1/N (L = {L}, K = {K})')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.show()
