import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kk_multithread import KKNetwork, train, single_update


if __name__ == "__main__":
    # Params
    L = 3
    K = 3
    N_values = [11, 21, 51, 101, 1001]
    labels = [f'N = {N}' for N in N_values]

    # Calculate average sync time for each N
    avg_sync_times = []
    for N in N_values:
        step_counts = train(L, N, K)
        avg_sync_time = np.mean(step_counts)
        avg_sync_times.append(avg_sync_time)

    # Plot with different colors for each point
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    for i, (N, color) in enumerate(zip(N_values, colors)):
        x = 1 / N
        y = avg_sync_times[i]
        plt.scatter(x, y, color=color, label=f'N={N}')
        plt.text(x, y, f'{y:.1f}', ha='left', va='bottom')

    # Manually set the x-axis range
    plt.xlim([-0.005, max([1/N for N in N_values]) + 0.012])

    # Add legend in the lower right corner
    plt.legend(loc='lower right')

    plt.xlabel('1/N')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time, L = {L}, K = {K}')
    plt.show()
