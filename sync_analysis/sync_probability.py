import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def simulate(
    L, K, N_values, num_runs=5000, zero_replace_1=1, zero_replace_2=-1,
    rule='anti_hebbian', state='anti_parallel'
):
    results = {}
    for N in N_values:
        print(f"Running N = {N}")
        sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        results[N] = sync_steps
    return results

def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    trials = len(sync_steps)
    if smooth:
        steps = np.linspace(1, max(sync_steps), num_intervals)  # Using linspace for fractional steps
    else:
        steps = np.percentile(np.sort(sync_steps), np.linspace(0, 100, num_intervals + 1))

    return [(step, sum(1 for s in sync_steps if s <= step) / trials) for step in steps]

def plot_sync_probs(simulation_results, L, K, num_intervals=20, smooth=True):
    colors = ['red', 'green', 'blue', 'black']
    markers = ['o', '^', 's', 'D']

    for (N, sync_steps), color, marker in zip(simulation_results.items(), colors, markers):
        scatter_probs = calculate_probs(sync_steps, num_intervals, smooth=False)
        smooth_probs = calculate_probs(sync_steps, num_intervals, smooth=smooth)
        filtered_scatter_probs = [(step, prob) for step, prob in scatter_probs if prob >= 0.65]
        if filtered_scatter_probs:
            plt.scatter(
                *zip(*filtered_scatter_probs),
                label=f'N = {N}', marker=marker, facecolors='none', edgecolors=color
            )

        plt.plot(*zip(*smooth_probs), linestyle='--', color=color, alpha=0.6)

        for step, prob in scatter_probs:
            if prob == 1.0:
                plt.text(int(step), 1.02, f'{int(step)}', ha='center', va='bottom', color=color)

    plt.xlabel('Steps')
    plt.ylabel('Sync Probability')
    y_ticks = np.linspace(0, 1, 11)
    plt.yticks(y_ticks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('./figures/transparent/t_sync_probability.png', transparent=True)
    plt.show()

def save_results_to_csv(simulation_results, file_path):
    df = pd.DataFrame(dict([(N, pd.Series(sync_steps)) for N, sync_steps in simulation_results.items()]))    
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    L, K = 3, 3
    N_values = [10, 16, 30, 1000]
    num_runs = 5000

    simulation_results = simulate(L, K, N_values, num_runs)

    plot_sync_probs(simulation_results, L, K)

    save_results_to_csv(simulation_results, './result/sync_results.csv')
