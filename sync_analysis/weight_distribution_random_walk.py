import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs

# parameters
L_values = [3, 4, 5, 6]  # Different values of L for the simulation
K = 3  # Number of hidden units
N = 100  # Number of input bits per hidden unit
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
num_runs = 10000  # Number of simulations to run
state = 'parallel'  # Synchronization state
learning_rule = 'random_walk'  # Learning rule to use
figure_file = './figures/transparent/weight_distribution_random_walk.png'  # File to save the figure

def calculate_weight_distribution(W, L):
    unique, counts = np.unique(W, return_counts=True)
    total_weights = W.size
    weight_distribution = {i: 0 for i in range(-L, L + 1)}
    for u, c in zip(unique, counts):
        if u in weight_distribution:
            weight_distribution[u] = c / total_weights
    return weight_distribution

if __name__ == '__main__':
    for L in L_values:
        x_values = list(range(-L, L + 1))  # x-axis for weights in range [-L, L]
        
        all_weight_distributions = {x: [] for x in x_values}

        print(f'Running random_walk with L={L}')
        results = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, learning_rule, state, return_weights=True)
        
        for steps, weights in results:
            weight_distribution = calculate_weight_distribution(weights, L)
            for x in x_values:
                all_weight_distributions[x].append(weight_distribution[x])

        avg_weight_distribution = [np.mean(all_weight_distributions[x]) for x in x_values]

        # Plot the weight distribution for this L
        plt.plot(x_values, avg_weight_distribution, marker='o', linestyle='-', label=f'L={L}')

        # Add uniform distribution line for each L
        uniform_distribution = [1 / (2 * L + 1)] * len(x_values)
        plt.plot(x_values, uniform_distribution, linestyle='--', color='gray', alpha=0.5)

        # Annotate y values below each point (for actual weight distribution)
        for i, y in enumerate(avg_weight_distribution):
            plt.text(x_values[i], y + 0.001, f'{100*y:.2f}', ha='center', va='bottom', fontsize=8)

        # Annotate the uniform distribution values on the right side of the line
        plt.text(x_values[-1] + 0.2, uniform_distribution[-1], f'{100*uniform_distribution[-1]:.2f}', ha='left', va='center', fontsize=8, color='gray')

    plt.xlabel('Weight values')
    plt.ylabel('Proportion')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    plt.savefig(figure_file, transparent=True)
    plt.show()
