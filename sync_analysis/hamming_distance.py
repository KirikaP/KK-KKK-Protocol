import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM

# parameters
L = 3  # Weight range [-L, L]
K = 3  # Number of hidden units
N = 100  # Number of input bits per hidden unit
num_runs = 5000  # Number of simulations to run
zero_replace_1 = -1  # Parameter for TPM initialization
zero_replace_2 = -1  # Parameter for TPM initialization
learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']  # Different learning rules
state = 'parallel'  # Synchronization state
max_workers = 16  # Maximum number of workers for parallel processing
output_file = './figures/hamming_distance_line_plot_parallel.png'  # File to save the figure

def calculate_hamming_distance(weights1, weights2):
    return np.sum(weights1 != weights2)

def run_single_simulation(L, K, N, zero_replace_1, zero_replace_2, rule, state):
    tpm1 = TPM(L, N, K, zero_replace_1)
    tpm2 = TPM(L, N, K, zero_replace_2)
    hamming_distances = []

    while True:
        X = np.random.choice([-1, 1], size=(K, N))

        tpm1.update_tau(X)
        tpm2.update_tau(X)

        if tpm1.tau == tpm2.tau:
            tpm1.update_W(X, rule)
            tpm2.update_W(X, rule)

            hamming_dist = calculate_hamming_distance(tpm1.W, tpm2.W)
            hamming_distances.append(hamming_dist)

            if tpm1.is_sync(tpm2, state):
                break

    return hamming_distances

def run_experiment(L, K, N, num_runs, rules, zero_replace_1=-1, zero_replace_2=-1, state='parallel', max_workers=None):
    avg_hamming_distances = {rule: [] for rule in rules}
    max_steps = 0

    for rule in rules:
        print(f'Running rule={rule}')
        all_distances = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_single_simulation, L, K, N, zero_replace_1, zero_replace_2, rule, state)
                for _ in range(num_runs)
            ]

            for future in tqdm(as_completed(futures), total=num_runs):
                hamming_distances = future.result()
                all_distances.append(hamming_distances)
                max_steps = max(max_steps, len(hamming_distances))

        avg_distances = np.zeros(max_steps)
        for distances in all_distances:
            distances = np.pad(distances, (0, max_steps - len(distances)), 'edge')
            avg_distances += distances
        avg_distances /= num_runs

        avg_hamming_distances[rule] = avg_distances

    return avg_hamming_distances

if __name__ == '__main__':
    avg_hamming_distances = run_experiment(L, K, N, num_runs, learning_rules, state=state, max_workers=max_workers)

    for rule in learning_rules:
        plt.plot(avg_hamming_distances[rule], label=f'{rule}')

    plt.xlabel('Steps')
    plt.ylabel('Average Hamming Distance')
    plt.yscale('log')
    plt.title('Hamming Distance over Time for Different Learning Rules')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.savefig(output_file, transparent=True)
    plt.show()
