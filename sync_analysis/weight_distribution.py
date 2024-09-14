import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def calculate_weight_distribution(W, L):
    unique, counts = np.unique(W, return_counts=True)
    total_weights = W.size
    weight_distribution = {i: 0 for i in range(-L, L+1)}
    for u, c in zip(unique, counts):
        if u in weight_distribution:
            weight_distribution[u] = c / total_weights
    return weight_distribution


if __name__ == '__main__':
    L, K, N = 5, 3, 100
    zero_replace_1, zero_replace_2 = -1, -1
    num_runs = 5000
    state = 'parallel'
    learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']
    
    x_values = list(range(-L, L + 1))  # x axis for weights in range [-L, L]
    
    all_weight_distributions = {rule: {x: [] for x in x_values} for rule in learning_rules}

    for rule in learning_rules:
        print(f'Running rule={rule}')
        results = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, return_weights=True)
        
        for steps, weights in results:
            weight_distribution = calculate_weight_distribution(weights, L)
            for x in x_values:
                all_weight_distributions[rule][x].append(weight_distribution[x])

    avg_weight_distributions = {rule: [np.mean(all_weight_distributions[rule][x]) for x in x_values] for rule in learning_rules}

    uniform_distribution = [1 / (2 * L + 1)] * len(x_values)

    for rule in learning_rules:
        plt.plot(x_values, avg_weight_distributions[rule], marker='o', linestyle='-', label=f'{rule}')

    # Plot the uniform distribution (horizontal line)
    plt.plot(x_values, uniform_distribution, linestyle='--', color='black', label='Random State', alpha=0.5)
    plt.xlabel('Weight values')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/transparent/weight_distribution.png')
    plt.show()
