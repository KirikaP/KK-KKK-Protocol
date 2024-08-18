import matplotlib.pyplot as plt
import numpy as np
from scripts.kk_multithread import train, KKNetwork


def calculate_probs(sync_steps, num_intervals=20, smooth=False):
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
    # Parameters
    np.random.seed(114)
    L = 3
    K = 3
    N_values = [10, 100, 1000]  # Different N values for testing
    colors = ['coral', 'green', 'black']
    labels = [f'N = {N}' for N in N_values]
    markers = ['^', 's', 'D']

    # Plotting
    plt.figure(figsize=(10, 6))
    for N, color, label, marker in zip(N_values, colors, labels, markers):
        # Initialize networks
        S = KKNetwork(L, N, K, zero_replacement=1)
        R = KKNetwork(L, N, K, zero_replacement=-1)
        
        # Train networks and get synchronization steps
        sync_steps = train(S, R, num_runs=5000)

        # Calculate interval-based probabilities for scatter plot
        scatter_probs = calculate_probs(sync_steps, smooth=False)
        # Calculate smooth probabilities for line plot
        smooth_probs = calculate_probs(sync_steps, smooth=True)

        # Extract steps and corresponding probabilities for scatter plot
        steps, probs = zip(*scatter_probs)
        # Extract steps and corresponding probabilities for smooth line plot
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
        plt.plot(smooth_steps, smooth_probs, linestyle='--', color=color, alpha=0.6)

    # Labels, title, legend, and grid
    plt.xlabel('Steps')
    plt.ylabel('Synchronization Probability')
    plt.title(f'Synchronization Probability vs. Steps (L = {L}, K = {K})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
