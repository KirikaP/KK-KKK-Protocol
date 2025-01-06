import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM
from tqdm import tqdm
import pandas as pd

# Parameters
K_values = [2, 3, 4, 5]         # Number of hidden units
N_values = [10, 100, 1000]      # Number of input bits per hidden unit
L_values = [1, 2, 3, 4, 5, 6]   # Weight range [-L, L]
rule = 'hebbian'                # Learning rule
sync_target = 'sender'          # Synchronization target
num_simulations = 5000          # Number of attack simulations
max_workers = 16                # Number of workers for parallel processing
csv_file = "C:/Users/hhh25/Desktop/dissertation/code/geometric_attack.csv"  # CSV file to save results

def initialize_tpms(L, N, K):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)
    attacker = TPM(L, N, K, -1)
    return sender, receiver, attacker

def geometric_attack(L, N, K, sync_target, rule):
    sender, receiver, attacker = initialize_tpms(L, N, K)
    target = sender if sync_target == 'sender' else receiver

    steps = 0

    while True:
        # Generate random input vector
        X = np.random.choice([-1, 1], size=(K, N))

        # Update tau for all TPMs
        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        # If sender and receiver are synchronized, update their weights
        if sender.tau == receiver.tau:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)

            # Simple attack
            if sender.tau == attacker.tau:
                attacker.update_W(X, rule=rule, tau_value=target.tau)
            # Geometric attack
            else:
                # Find the index of the hidden unit with the minimum absolute inner product
                min_h_index = np.argmin(np.abs(attacker.h))
                # Flip the sign of the corresponding sigma for the attacker
                attacker.sigma[min_h_index] = -attacker.sigma[min_h_index]
                # Update the weights of the attacker
                attacker.update_W(X, rule=rule, tau_value=target.tau)

        # Check if attacker and sender are synchronized
        if attacker.is_sync(sender, state='parallel'):
            return True  # Return true if attacker syncs with sender

        # Check if sender and receiver are synchronized
        if sender.is_sync(receiver, state='parallel'):
            return False  # Return false if sender and receiver sync first

        steps += 1

def attacker_learn(L, N, K, sync_target, rule, num_simulations, max_workers):
    successes = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(geometric_attack, L, N, K, sync_target, rule)
            for _ in range(num_simulations)
        ]

        # Add tqdm progress bar for the simulations
        for future in tqdm(as_completed(futures), total=num_simulations, desc=f'Running L={L}, N={N}, K={K}'):
            success = future.result()
            if success:
                successes += 1

    # Calculate the probability of success
    success_probability = successes / num_simulations
    return success_probability


if __name__ == "__main__":
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file)
    else:
        existing_data = pd.DataFrame(columns=['K', 'N', 'L', 'Success Probability (%)'])

    results = []

    total_combinations = len(K_values) * len(N_values) * len(L_values)
    
    with tqdm(total=total_combinations, desc="Total Progress") as total_pbar:
        for K in K_values:
            for N in N_values:
                for L in L_values:
                    # Check if the simulation has already been run
                    if not ((existing_data['K'] == K) & (existing_data['N'] == N) & (existing_data['L'] == L)).any():
                        success_probability = attacker_learn(L, N, K, sync_target, rule, num_simulations, max_workers)
                        success_percentage = success_probability * 100
                        print(f"K={K}, N={N}, L={L}: {success_percentage:.2f}%")
                        results.append([K, N, L, success_percentage])
                    else:
                        print(f"Skipping for K={K}, N={N}, L={L}")
                    
                    total_pbar.update(1)

    # Append the new results to the existing data and save to the CSV file
    if results:
        new_data = pd.DataFrame(results, columns=['K', 'N', 'L', 'Success Probability (%)'])
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(csv_file, index=False)
        print(f"New results added to {csv_file}")
