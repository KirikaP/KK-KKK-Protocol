import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import entropy
import pandas as pd  # Import pandas to save data as CSV
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs


def calculate_weight_entropy(W):
    unique, counts = np.unique(W, return_counts=True)
    probabilities = counts / W.size
    return entropy(probabilities, base=2)

def calculate_uniform_entropy(L):
    return np.log2(2 * L + 1)

def save_entropy_to_csv(L_values, all_avg_entropy, uniform_entropies, file_path):
    # Create a DataFrame to store the entropy values
    df = pd.DataFrame({'L': L_values, 'Uniform Entropy': uniform_entropies})
    
    # Add the average entropies for each rule to the DataFrame
    for rule, entropies in all_avg_entropy.items():
        df[rule] = entropies
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == '__main__':
    K, N = 3, 100
    zero_replace_1, zero_replace_2 = -1, -1
    num_runs = 2000
    state = 'parallel'
    learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']
    L_values = range(4, 7)
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

    plt.xlabel('L')
    plt.ylabel('Average Entropy')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('./figures/transparent/weight_entropy.png', transparent=True)
    plt.show()

    save_entropy_to_csv(L_values, all_avg_entropy, uniform_entropies, './result/entropy.csv')
