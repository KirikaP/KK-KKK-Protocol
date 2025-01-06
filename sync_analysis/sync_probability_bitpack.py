import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs

# parameters
L = 3  # Weight range [-L, L]
K = 3  # Number of hidden units
N = 100  # Number of input bits per hidden unit
num_runs = 5000  # Number of simulations to run
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
state = 'parallel'  # Synchronization state
rules = ['hebbian', 'anti_hebbian', 'random_walk']  # Different learning rules
bit_package = 8  # Bit package size for 'anti_hebbian' rule
output_file = './result/sync_results_N100_with_anti_hebbian_B8.csv'  # CSV file to save results
figure_file = './figures/transparent/t_sync_probability.png'  # File to save the figure

def simulate(
    L, K, N, rule, num_runs=5000, zero_replace_1=-1, zero_replace_2=-1, state='parallel', B=None
):
    if B is not None:
        print(f"Running N = {N} with rule = {rule}, B = {B}")
    else:
        print(f"Running N = {N} with rule = {rule}")
    sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, B=B)
    return sync_steps

def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    trials = len(sync_steps)
    if smooth:
        steps = np.linspace(1, max(sync_steps), num_intervals)  # Using linspace for fractional steps
    else:
        steps = np.percentile(np.sort(sync_steps), np.linspace(0, 100, num_intervals + 1))

    return [(step, sum(1 for s in sync_steps if s <= step) / trials) for step in steps]

def plot_sync_probs(simulation_results, L, K, num_intervals=20, smooth=True):
    colors = ['red', 'green', 'blue', 'purple']
    markers = ['o', '^', 's', 'd']
    labels = list(simulation_results.keys())

    for label, color, marker in zip(labels, colors, markers):
        sync_steps = simulation_results[label]
        scatter_probs = calculate_probs(sync_steps, num_intervals, smooth=False)
        smooth_probs = calculate_probs(sync_steps, num_intervals, smooth=smooth)
        
        plt.scatter(
            *zip(*scatter_probs),
            label=f'{label}', marker=marker, facecolors='none', edgecolors=color
        )

        plt.plot(*zip(*smooth_probs), linestyle='--', color=color, alpha=0.6)

        for step, prob in scatter_probs:
            if prob == 1.0:
                plt.text(int(step), 1.02, f'{int(step)}', ha='center', va='bottom', color=color)

    plt.xlabel('Steps')
    plt.ylabel('Sync Probability')
    y_ticks = np.linspace(0, 1, 11)
    plt.yticks(y_ticks)
    plt.ylim(0, 1.1)  # 将y轴最大值设置为1.1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    plt.savefig(figure_file, transparent=True)
    plt.show()

def save_results_to_csv(simulation_results, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame(dict([(label, pd.Series(sync_steps)) for label, sync_steps in simulation_results.items()]))    
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    simulation_results = {}

    for rule in rules:
        simulation_results[rule] = simulate(L, K, N, rule, num_runs)

    # Run simulation for 'anti_hebbian' rule with B=8
    simulation_results[f'anti_hebbian_B={bit_package}'] = simulate(L, K, N, 'anti_hebbian', num_runs, B=bit_package)

    # Plot synchronization probabilities
    plot_sync_probs(simulation_results, L, K)

    # Save the results to a CSV file
    save_results_to_csv(simulation_results, output_file)
