import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs

# parameters
K = 3  # Number of hidden units
N = 100  # Number of input bits per hidden unit
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
num_runs = 2000  # Number of simulations to run
state = 'parallel'  # Synchronization state
learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']  # Different learning rules
L_values = range(1, 7)  # Weight range values [-L, L]
output_file = './result/boundary_probability.csv'  # CSV file to save results
figure_file = './figures/transparent/weight_boundary_probability.png'  # File to save the figure

def calculate_boundary_probability(W, L):
    boundary_weights = np.sum((W == -L) | (W == L))
    total_weights = W.size
    return boundary_weights / total_weights

def save_results_to_csv(L_values, all_avg_boundary_probs, theoretical_boundary_probs, file_path):
    df = pd.DataFrame({'L': L_values, 'Theoretical': theoretical_boundary_probs})

    for rule, boundary_probs in all_avg_boundary_probs.items():
        df[rule] = boundary_probs

    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == '__main__': 
    all_avg_boundary_probs = {rule: [] for rule in learning_rules}

    for rule in learning_rules:
        for L in L_values:
            print(f'Running L={L} and rule={rule}')
            results = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, return_weights=True)
            boundary_probs = [calculate_boundary_probability(weights, L) for steps, weights in results]
            avg_boundary_prob = np.mean(boundary_probs)
            all_avg_boundary_probs[rule].append(avg_boundary_prob)

    # Theoretical boundary probabilities
    theoretical_boundary_probs = [2 / (2 * L + 1) for L in L_values]

    # Plotting
    for rule in learning_rules:
        plt.plot(L_values, all_avg_boundary_probs[rule], marker='o', linestyle='-', label=f'{rule}')

    plt.plot(L_values, theoretical_boundary_probs, marker='x', linestyle='--', color='black', label='Random State', alpha=0.5)

    plt.xlabel('L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(figure_file, transparent=True)
    plt.show()

    # Save results to CSV
    save_results_to_csv(L_values, all_avg_boundary_probs, theoretical_boundary_probs, output_file)
