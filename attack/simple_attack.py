import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM

# Constants for the simulation
L, K = 3, 3  # TPM parameters
RULE = 'hebbian'  # Learning rule
SYNC_TARGET = 'sender'  # Synchronization target ('sender' or 'receiver')
NUM_SIMULATIONS = 2000  # Number of attack simulations
MAX_WORKERS = 16  # Number of workers for parallel processing
BIN_WIDTH = 0.01  # Histogram bin width
N_VALUES = [100, 50, 20, 10]  # Different N values to evaluate

def initialize_tpms(L, N, K):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)
    attacker = TPM(L, N, K, -1)
    return sender, receiver, attacker

def attack_step(L, N, K, sync_target, rule):
    sender, receiver, attacker = initialize_tpms(L, N, K)
    target = sender if sync_target == 'sender' else receiver
    
    steps, sync_steps, attack_sync_steps = 0, None, None

    while True:
        # Generate random input vector
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))
        
        # Update tau for all TPMs
        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        # Update weights if sender and receiver are synchronized
        if sender.tau * receiver.tau > 0:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
            attacker.update_W(X, rule=rule, tau_value=target.tau)

        # Check if sender and receiver are synchronized
        if sync_steps is None and sender.is_sync(receiver, state='parallel'):
            sync_steps = steps

        # Check if attacker and target (sender or receiver) are synchronized
        if attack_sync_steps is None and attacker.is_sync(target, state='parallel'):
            attack_sync_steps = steps

        # If both are synchronized, break
        if sync_steps is not None and attack_sync_steps is not None:
            break

        steps += 1

    # Return the ratio of sync_steps to attack_sync_steps
    return sync_steps / attack_sync_steps if attack_sync_steps is not None else None

def attacker_learn(L, N, K, sync_target, rule, num_simulations, max_workers):
    ratios = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(attack_step, L, N, K, sync_target, rule)
            for _ in range(num_simulations)
        ]

        for future in tqdm(as_completed(futures), total=num_simulations):
            ratio = future.result()
            if ratio is not None:
                ratios.append(ratio)

    return ratios

def plot_results(ratios_dict, bin_width=BIN_WIDTH):
    for N, ratios in ratios_dict.items():
        truncated_ratios = [r for r in ratios if r < 1.0]
        
        if N == 20:
            plt.hist(truncated_ratios, bins=np.arange(0, 1 + 0.005, bin_width), alpha=0.5, histtype='step', edgecolor='black', label=f'N = {N}')
        else:
            plt.hist(truncated_ratios, bins=np.arange(0, 1 + 0.005, bin_width), alpha=0.5, label=f'N = {N}')
    
    plt.xlabel('Ratio(r) (sync_steps / attack_sync_steps)')
    plt.ylabel('P(r)')
    plt.title('Distribution between Ratio(r) and N Values')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="N Values")
    plt.show()

if __name__ == "__main__":
    # Dictionary to store ratios for different N values
    ratios_dict = {}

    for N in N_VALUES:
        print(f"Running N = {N}...")
        ratios = attacker_learn(L, N, K, SYNC_TARGET, RULE, NUM_SIMULATIONS, MAX_WORKERS)
        ratios_dict[N] = ratios

    # Plot the results
    plot_results(ratios_dict)
