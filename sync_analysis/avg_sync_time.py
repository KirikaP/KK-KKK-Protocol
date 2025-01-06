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
N_values = [10, 100, 1000, 10000]  # Different values of N for the simulation
num_runs = 3000  # Number of simulations to run
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']  # Different learning rules
state = 'parallel'  # Synchronization state
output_file = './result/avg_sync_time.csv'  # CSV file to save results
figure_file = './figures/transparent/t_sync_with_N.png'  # File to save the figure

def simulate(
    L, K, N_values, num_runs=5000, zero_replace_1=-1, zero_replace_2=-1,
    rule='anti_hebbian', state='parallel'
):
    results = []
    for N in N_values:
        print(f"Running N = {N} for {rule}")
        sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state)
        mean_steps = np.mean(sync_steps)
        results.append(mean_steps)

    return results

def plot_results(N_values, results_dict):
    plt.figure()

    # Define markers for different rules
    markers = {
        'hebbian': 'o',          # Hollow circle
        'anti_hebbian': 's',     # Hollow square
        'random_walk': '^'       # Hollow triangle
    }

    # Plot the results for each rule with hollow markers
    for rule, avg_sync_times in results_dict.items():
        for N, y in zip(N_values, avg_sync_times):
            x = np.log10(N)  # Log scale on the x-axis
            plt.scatter(x, y, edgecolor='black', facecolor='none', marker=markers[rule], label=rule if N == N_values[0] else "")

    y_max = max([max(values) for values in results_dict.values()])
    plt.ylim([0, y_max + 0.1 * y_max])

    # Logarithmic labels for x-axis
    plt.xlabel(r'$\log_{10}{N}$')
    plt.ylabel('Average Synchronization Steps')
    plt.grid(True, alpha=0.5, linestyle='--')

    # Add a legend to differentiate learning rules
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    # Save the figure
    plt.savefig(figure_file, transparent=True)
    plt.show()

def save_results_to_csv(N_values, results_dict, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame({
        'N': N_values,
        'Hebbian': results_dict['hebbian'],
        'Anti-Hebbian': results_dict['anti_hebbian'],
        'Random Walk': results_dict['random_walk']
    })
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    # Dictionary to store the results for each learning rule
    results_dict = {}

    # Simulate for each learning rule
    for rule in learning_rules:
        avg_sync_times = simulate(L, K, N_values, num_runs, zero_replace_1, zero_replace_2, rule, state)
        results_dict[rule] = avg_sync_times

    # Plot the results with log10 scaling on x-axis and hollow markers
    plot_results(N_values, results_dict)

    # Save the results to a CSV file
    save_results_to_csv(N_values, results_dict, output_file)
