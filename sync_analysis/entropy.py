import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import entropy
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def calculate_weight_entropy(W):
    unique, counts = np.unique(W, return_counts=True)
    probabilities = counts / W.size
    return entropy(probabilities, base=2)

def calculate_uniform_entropy(L):
    return np.log2(2 * L + 1)


if __name__ == '__main__':
    K, N = 3, 100
    zero_replace_1, zero_replace_2 = -1, -1
    num_runs = 500
    state = 'parallel'
    learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']
    L_values = range(1, 7)
    all_avg_entropy = {rule: [] for rule in learning_rules}

    for rule in learning_rules:
        for L in L_values:
            print(f'Running L={L} and rule={rule}')
            results = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, return_weights=True)
            entropies = [calculate_weight_entropy(weights) for steps, weights in results]
            avg_entropy = np.mean(entropies)
            all_avg_entropy[rule].append(avg_entropy)

    # Calculate theoretical entropy for uniform distribution
    uniform_entropies = [calculate_uniform_entropy(L) for L in L_values]

    # Plot learning rule entropies
    for rule in learning_rules:
        plt.plot(L_values, all_avg_entropy[rule], marker='o', linestyle='-', label=f'{rule}')

    # Plot the uniform entropy curve
    plt.plot(L_values, uniform_entropies, marker='x', linestyle='--', color='black', label='Uniform Entropy', alpha=0.5)

    plt.xlabel('L values')
    plt.ylabel('Average Entropy')
    plt.title('Average Weight Entropy for L and Learning Rules')
    plt.legend()
    plt.grid(True)
    plt.show()
