import matplotlib.pyplot as plt
import numpy as np
from scripts.parity_machine import TreeParityMachine as TPM, train_TPMs


def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    """
    Calculate the probability of synchronization for each step interval.

    Args:
        sync_steps (list): List of synchronization step counts.
        num_intervals (int): Number of intervals to divide the steps into (only used if smooth=False).
        smooth (bool): If True, calculate probabilities for each step up to max(sync_steps).

    Returns:
        List of (step, probability) tuples.
    """
    trials = len(sync_steps)
    if smooth:
        steps = range(1, max(sync_steps) + 1)
    else:
        steps = np.percentile(np.sort(sync_steps), np.linspace(0, 100, num_intervals + 1))

    return [(step, sum(1 for s in sync_steps if s <= step) / trials) for step in steps]


def plot_sync_probs(N_values, L, K, num_runs=5000, num_intervals=20, smooth=True):
    """
    Plot the synchronization probability vs. steps for multiple N values.

    Args:
        N_values (list): List of N values.
        L (int): Weight limit range [-L, L].
        K (int): Number of hidden units.
        num_runs (int): Number of runs for each N value.
        num_intervals (int): Number of intervals for probability calculation (if smooth=False).
        smooth (bool): Whether to smooth the probability over all steps.
    """
    colors = ['red', 'green', 'blue', 'black']
    markers = ['o', '^', 's', 'D']

    for N, color, marker in zip(N_values, colors, markers):
        print(f"Running simulation for N = {N}")
        sync_steps = train_TPMs(L, K, N, 1, -1, num_runs)

        # Calculate probabilities
        scatter_probs = calculate_probs(sync_steps, num_intervals, smooth=False)
        smooth_probs = calculate_probs(sync_steps, smooth=smooth)

        # Plot scatter points where probability >= 0.65
        filtered_scatter_probs = [(step, prob) for step, prob in scatter_probs if prob >= 0.65]
        if filtered_scatter_probs:
            plt.scatter(*zip(*filtered_scatter_probs), label=f'N = {N}', marker=marker, facecolors='none', edgecolors=color)

        # Plot the smooth probability curve
        plt.plot(*zip(*smooth_probs), linestyle='--', color=color, alpha=0.6)

        # Draw a vertical line at the step where probability reaches 1.0
        for step, prob in scatter_probs:
            if prob == 1.0:
                plt.axvline(x=step, color=color, linestyle=':', alpha=0.6)
                plt.text(step, 1.02, f'{step}', ha='center', va='bottom', color=color)

    plt.xlabel('Steps')
    plt.ylabel('Sync Probability')
    plt.title(f'Sync Probability vs. Steps (L = {L}, K = {K})')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Parameters
    L, K = 3, 3
    N_values = [10, 16, 30, 1000]
    num_runs = 5000  # Number of runs for each N value

    # Run and plot synchronization probability for different N values
    plot_sync_probs(N_values, L, K, num_runs)
    