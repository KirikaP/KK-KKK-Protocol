import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM

# parameters
L = 3  # Weight range [-L, L]
K = 3  # Number of hidden units
rule = 'hebbian'  # Learning rule
sync_target = 'sender'  # Synchronization target
num_simulations = 2000  # Number of attack simulations
max_workers = 16  # Number of workers for parallel processing
N_values = [100, 50, 20, 10]  # Different values of N for the simulation
output_file = "simple_attack.csv"  # Output file for results

def initialize_tpms(L, N, K):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)
    attacker = TPM(L, N, K, -1)
    return sender, receiver, attacker

def simple_attack(L, N, K, sync_target, rule):
    sender, receiver, attacker = initialize_tpms(L, N, K)
    target = sender if sync_target == 'sender' else receiver

    steps, sync_steps, attack_sync_steps = 0, None, None

    while True:
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))
        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        if sender.tau == receiver.tau:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
            attacker.update_W(X, rule=rule, tau_value=target.tau)

        if sync_steps is None and sender.is_sync(receiver, state='parallel'):
            sync_steps = steps

        if attack_sync_steps is None and attacker.is_sync(target, state='parallel'):
            attack_sync_steps = steps

        if sync_steps is not None and attack_sync_steps is not None:
            break

        steps += 1

    return sync_steps / attack_sync_steps if attack_sync_steps is not None else None

def attacker_learn(L, N, K, sync_target, rule, num_simulations, max_workers):
    ratios = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simple_attack, L, N, K, sync_target, rule)
            for _ in range(num_simulations)
        ]

        for future in tqdm(as_completed(futures), total=num_simulations):
            ratio = future.result()
            if ratio is not None:
                ratios.append(ratio)

    return ratios

if __name__ == "__main__":
    results = []

    for N in N_values:
        print(f"Running N = {N}...")
        ratios = attacker_learn(L, N, K, sync_target, rule, num_simulations, max_workers)

        success_rate = sum(1 for r in ratios if r >= 1.0) / len(ratios)  # Calculate success rate
        for ratio in ratios:
            results.append([L, N, K, ratio, success_rate])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV file
    df = pd.DataFrame(results, columns=["L", "N", "K", "Ratio", "Success Rate"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
