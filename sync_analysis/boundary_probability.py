import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


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
    K, N = 3, 100
    zero_replace_1, zero_replace_2 = -1, -1
    num_runs = 2000
    state = 'parallel'
    learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']
    L_values = range(1, 7)
    all_avg_boundary_probs = {rule: [] for rule in learning_rules}

    for rule in learning_rules:
        for L in L_values:
            print(f'Running L={L} and rule={rule}')
            results = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, return_weights=True)
            boundary_probs = [calculate_boundary_probability(weights, L) for steps, weights in results]
            avg_boundary_prob = np.mean(boundary_probs)
            all_avg_boundary_probs[rule].append(avg_boundary_prob)

    theoretical_boundary_probs = [2 / (2 * L + 1) for L in L_values]

    for rule in learning_rules:
        plt.plot(L_values, all_avg_boundary_probs[rule], marker='o', linestyle='-', label=f'{rule}')

    plt.plot(L_values, theoretical_boundary_probs, marker='x', linestyle='--', color='black', label='Theoretical', alpha=0.5)

    plt.xlabel('L')
    plt.ylabel('Average Boundary Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/transparent/weight_boundary_probability.png', transparent=True)
    plt.show()

    save_results_to_csv(L_values, all_avg_boundary_probs, theoretical_boundary_probs, './result/boundary_probability.csv')
