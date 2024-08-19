import matplotlib.pyplot as plt
import numpy as np
from scripts.kk_multithread import train


def simulate(L, K, N_values):
    return [np.mean(train(L, N, K)) for N in N_values]


if __name__ == "__main__":
    np.random.seed(114)
    L, K, N_values = 3, 3, [10, 17, 31, 100, 316, 1000]
    avg_sync_times = simulate(L, K, N_values)

    for N, y in zip(N_values, avg_sync_times):
        x = np.log10(N)
        plt.scatter(
            x, y,
            edgecolors='black',
            facecolors='none',
            label=f'N={N}'
        )
        plt.text(
            x, y,
            f'{y:.2f}',
            ha='left',
            va='bottom'
        )

    plt.xlim([np.log10(min(N_values)) - 0.2, np.log10(max(N_values)) + 0.2])
    plt.xlabel('log_10(N)')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time vs N (L = {L}, K = {K})')
    plt.grid(True)
    plt.show()
