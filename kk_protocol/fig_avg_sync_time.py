import matplotlib.pyplot as plt
import numpy as np
from scripts.kk_multithread import train, KKNetwork


if __name__ == "__main__":
    # Parameters
    np.random.seed(114)
    L = 3
    K = 3
    # N_values = [10, 32, 100, 316, 1000, 3162, 10000]  # Different N values
    N_values = [100, 316, 1000, 3162, 10000]
    labels = [f'N = {N}' for N in N_values]

    # Calculate average sync time for each N
    avg_sync_times = []
    for N in N_values:
        S = KKNetwork(L, N, K, zero_replacement=1)
        R = KKNetwork(L, N, K, zero_replacement=-1)
        step_counts = train(S, R)
        avg_sync_time = np.mean(step_counts)
        avg_sync_times.append(avg_sync_time)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(N_values)))  # Generate a rainbow color map

    for i, (N, color) in enumerate(zip(N_values, colors)):
        x = np.log10(N)  # Use log10(N) for x-axis
        y = avg_sync_times[i]
        plt.scatter(x, y, color=color, label=f'N={N}', s=100)  # Larger scatter points
        plt.text(x, y, f'{y:.1f}', ha='left', va='bottom', fontsize=9)  # Annotate points

    # Manually set the x-axis range to match the log scale
    plt.xlim([np.log10(min(N_values)) - 0.2, np.log10(max(N_values)) + 0.2])

    # Add legend
    plt.legend(loc='upper left', fontsize=10)

    # Labels and Title
    plt.xlabel('log_10(N)')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time vs N (L = {L}, K = {K})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
