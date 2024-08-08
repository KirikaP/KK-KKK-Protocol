import matplotlib.pyplot as plt
import numpy as np
from kk_multithread import train


def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    # num_intervals: number of intervals for interval-based scatter plot
    trials = len(sync_steps)
    max_step = max(sync_steps)
    sorted_steps = np.sort(sync_steps)

    if smooth:
        # For smooth probabilities, calculate probability at every step
        steps = range(1, max_step + 1)
    else:
        # For interval-based scatter plot
        steps = np.percentile(sorted_steps, np.linspace(0, 100, num_intervals + 1))

    probs = []
    for step in steps:
        successful_syncs = sum(1 for s in sync_steps if s <= step)
        probability = successful_syncs / trials
        probs.append((step, probability))

    return probs


if __name__ == "__main__":
    # params
    L = 3
    K = 3
    N_values = [11, 101, 1001]
    colors = ['coral', 'green', 'black']
    labels = [f'N = {N}' for N in N_values]
    markers = ['^', 's', 'D']

    # plot
    for N, color, label, marker in zip(N_values, colors, labels, markers):
        sync_steps = train(L, N, K, num_runs=5000)
        # Calculate interval-based probs for scatter plot
        scatter_probs = calculate_probs(sync_steps, smooth=False)
        # Calculate smooth probs for line plot
        smooth_probs = calculate_probs(sync_steps, smooth=True)

        # Extract steps and corresponding probs for scatter plot
        steps, probs = zip(*scatter_probs)
        # Extract steps and corresponding probs for smooth line plot
        smooth_steps, smooth_probs = zip(*smooth_probs)

        # Plot probability scatter
        plt.scatter(
            steps,
            probs,
            label=label,
            color=color,
            marker=marker,
            facecolors='none',
            edgecolors=color
        )
        # Plot smooth line
        plt.plot(smooth_steps, smooth_probs, linestyle='--', color=color, alpha=0.3)

    plt.xlabel('Steps')
    plt.ylabel('Synchronization Probability')
    plt.title(f'Synchronization Probability vs. Steps, L = {L}, K = {K}')
    plt.legend()
    plt.grid(True)
    plt.show()
