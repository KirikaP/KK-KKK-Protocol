import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def simulate(
    L, K, N_values, num_runs=5000, zero_replace_1=1, zero_replace_2=-1,
    rule='anti_hebbian', state='anti_parallel'
):
    """
    Run simulations for multiple N values

    Args:
        L (int): Weight limit range [-L, L]
        K (int): Number of hidden units
        N_values (list): List of N values to simulate
        num_runs (int): Number of runs per N value
        zero_replace_1 (int): Zero replacement for the first TPM
        zero_replace_2 (int): Zero replacement for the second TPM
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk')
        state (str): Synchronization state ('parallel' or 'anti_parallel')

    Returns:
        dict: Dictionary with N values as keys and list of sync steps as values
    """
    results = {}
    for N in N_values:
        print(f"Running N = {N}")
        sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        results[N] = sync_steps
    return results

def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    """
    Calculate the probability of synchronization for each step interval

    Args:
        sync_steps (list): List of synchronization step counts
        num_intervals (int): Number of intervals to divide the steps into (only used if smooth=False)
        smooth (bool): If True, calculate probabilities for each step up to max(sync_steps)

    Returns:
        List of (step, probability) tuples
    """
    trials = len(sync_steps)
    if smooth:
        steps = range(1, max(sync_steps) + 1)
    else:
        steps = np.percentile(np.sort(sync_steps), np.linspace(0, 100, num_intervals + 1))

    return [(step, sum(1 for s in sync_steps if s <= step) / trials) for step in steps]

def plot_sync_probs(simulation_results, L, K, num_intervals=20, smooth=True):
    """
    Plot the synchronization probability vs. steps for multiple N values

    Args:
        simulation_results (dict): Dictionary with N values as keys and list of sync steps as values
        L (int): Weight limit range [-L, L]
        K (int): Number of hidden units
        num_intervals (int): Number of intervals for probability calculation (if smooth=False)
        smooth (bool): Whether to smooth the probability over all steps
    """
    colors = ['red', 'green', 'blue', 'black']
    markers = ['o', '^', 's', 'D']

    for (N, sync_steps), color, marker in zip(simulation_results.items(), colors, markers):
        print(f"Plotting for N = {N}")

        # Calculate probabilities
        scatter_probs = calculate_probs(sync_steps, num_intervals, smooth=False)
        smooth_probs = calculate_probs(sync_steps, smooth=smooth)

        # Plot scatter points where probability >= 0.65
        filtered_scatter_probs = [(step, prob) for step, prob in scatter_probs if prob >= 0.65]
        if filtered_scatter_probs:
            plt.scatter(
                *zip(*filtered_scatter_probs),
                label=f'N = {N}', marker=marker, facecolors='none', edgecolors=color
            )

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
    # Simulation parameters
    L, K = 3, 3
    N_values = [10, 16, 30, 1000]
    num_runs = 5000

    # Run the simulations
    simulation_results = simulate(L, K, N_values, num_runs)

    # Plot the results
    plot_sync_probs(simulation_results, L, K)
