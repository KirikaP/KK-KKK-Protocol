import matplotlib.pyplot as plt
import numpy as np
from kk_multithread import train


def calculate_probabilities(sync_steps, num_intervals=20):
    probabilities = []
    trials = len(sync_steps)

    # Sort synchronization steps
    sorted_steps = np.sort(sync_steps)

    # Determine interval points
    interval_steps = np.percentile(sorted_steps, np.linspace(0, 100, num_intervals + 1))

    for step in interval_steps:
        successful_syncs = sum(1 for s in sync_steps if s <= step)
        probability = successful_syncs / trials
        probabilities.append((step, probability))
    
    return probabilities


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
        # Get synchronization steps for each trial
        sync_steps = train(L, N, K, num_runs=5000)
        # Calculate probabilities
        probabilities = calculate_probabilities(sync_steps)
        # Extract steps and corresponding probabilities
        steps, probs = zip(*probabilities)
        # Plot probability scatter
        plt.scatter(steps, probs, label=label, color=color, marker=marker,
                    facecolors='none', edgecolors=color, alpha=0.7)

    plt.xlabel('Steps')
    plt.ylabel('Synchronization Probability')
    plt.title(f'Synchronization Probability vs. Steps, L = {L}, K = {K}')
    plt.legend()
    plt.grid(True)
    plt.show()
