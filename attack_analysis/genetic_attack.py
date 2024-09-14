import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM
from tqdm import tqdm
from itertools import product
import csv

# Parameters
K_values = [3]         # Number of units
N_values = [100]   # Input size
L_values = [1, 2, 3, 4, 5]   # Weight range
M_values = [2500]  # Number of attackers
rule = 'hebbian'             # Learning rule
sync_target = 'sender'       # Target of the attack (sender or receiver)
num_simulations = 2000       # Number of simulations per parameter combination

def gen_sigma(K, tau):
    num_sigma_comb = 2 ** (K - 1)
    sigma_combinations = np.ones((num_sigma_comb, K), dtype=int)  # Initialize with all ones

    for i in range(num_sigma_comb):
        bin_rep = np.array(list(bin(i)[2:].zfill(K - 1)), dtype=int)
        bin_rep = np.where(bin_rep == 1, -1, 1)
        sigma_combinations[i, :K - 1] = bin_rep
        sigma_combinations[i, K - 1] = int(tau * np.prod(sigma_combinations[i, :K - 1]))

    return sigma_combinations

def genetic_attack(L, N, K, M, sync_target, rule):
    num_sigma_comb = 2 ** (K - 1)
    sigma_pos_tau = gen_sigma(K, 1)
    sigma_neg_tau = gen_sigma(K, -1)

    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)

    # Initialize the weights of the attacker swarm
    attacker_swarm_W = np.random.choice(np.arange(-L, L + 1), size=(1, K, N))

    target = sender if sync_target == 'sender' else receiver

    steps = 0
    attack_success = False

    while True:
        steps += 1

        # Generate random input vectors
        X = np.random.choice([-1, 1], size=(K, N))

        # Update tau for sender and receiver
        sender.update_tau(X)
        receiver.update_tau(X)

        # Attacker swarm updates
        attacker_swarm_sigma = np.sign(np.sum(attacker_swarm_W * X, axis=2, keepdims=True))
        attacker_swarm_sigma = np.where(attacker_swarm_sigma == 0, -1, attacker_swarm_sigma)
        attacker_swarm_tau = np.prod(attacker_swarm_sigma, axis=1, keepdims=True)

        if sender.tau == receiver.tau:
            # Update weights for sender and receiver
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)

            Q = attacker_swarm_W.shape[0]

            if Q < M:
                attacker_swarm_W_variants = np.repeat(attacker_swarm_W, repeats=num_sigma_comb, axis=0)
                sigma_combinations = sigma_pos_tau if target.tau == 1 else sigma_neg_tau

                for i in range(Q):
                    for j in range(num_sigma_comb):
                        idx = i * num_sigma_comb + j
                        for row in range(K):
                            if sigma_combinations[j][row] == sender.tau:
                                attacker_swarm_W_variants[idx, row, :] += sender.tau * X[row, :]
                attacker_swarm_W = attacker_swarm_W_variants
            else:
                matching_indices = np.where(attacker_swarm_tau.flatten() == sender.tau)[0]
                attacker_swarm_W = attacker_swarm_W[matching_indices]
                attacker_swarm_sigma = attacker_swarm_sigma[matching_indices]
                attacker_swarm_tau = attacker_swarm_tau[matching_indices]

                for idx in range(attacker_swarm_W.shape[0]):
                    for row in range(K):
                        if attacker_swarm_sigma[idx, row, :] == attacker_swarm_tau[idx]:
                            attacker_swarm_W[idx, row, :] += attacker_swarm_tau[idx].item() * X[row, :]

        attacker_swarm_W = np.clip(attacker_swarm_W, -L, L)

        # Check for successful attack
        for idx in range(attacker_swarm_W.shape[0]):
            if np.array_equal(attacker_swarm_W[idx], sender.W):
                attack_success = True
                return attack_success, steps

        # Check if sender and receiver are synchronized
        if sender.is_sync(receiver, state='parallel'):
            break

    return attack_success, steps

def run_simulation(L, N, K, M, sync_target, rule, num_simulations):
    success_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(genetic_attack, L, N, K, M, sync_target, rule) for _ in range(num_simulations)
        ]

        # Initialize tqdm progress bar
        with tqdm(total=num_simulations, desc=f'K={K}, N={N}, L={L}, M={M}', leave=False) as pbar:
            for future in as_completed(futures):
                attack_success, _ = future.result()
                if attack_success:
                    success_count += 1
                pbar.update(1)  # Update progress bar

    return success_count


if __name__ == '__main__':
    # Open the CSV file to write the results
    with open('genetic_attack.csv', 'w', newline='') as csvfile:
        # Define the CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['K', 'N', 'L', 'M', 'Success Rate (%)'])

        # Total number of parameter combinations
        total_combinations = len(K_values) * len(N_values) * len(L_values) * len(M_values)

        # Initialize overall progress bar
        with tqdm(total=total_combinations, desc='Total Progress') as total_pbar:
            # Loop over all parameter combinations
            for K, N, L, M in product(K_values, N_values, L_values, M_values):
                # Print message moved inside tqdm description
                # Run simulations with progress bar
                success_count = run_simulation(L, N, K, M, sync_target, rule, num_simulations)

                # Calculate success rate
                success_rate = (success_count / num_simulations) * 100

                # Write the results to the CSV file
                csv_writer.writerow([K, N, L, M, f'{success_rate:.2f}'])

                # Flush the file to ensure data is written
                csvfile.flush()

                # Update overall progress bar
                total_pbar.update(1)
