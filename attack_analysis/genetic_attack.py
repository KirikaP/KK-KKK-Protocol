import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM

# Parameter settings
L = 3  # Weight range
N = 100  # Input size
K = 2  # Number of perceptrons
M = 2500  # Maximum number of attackers
rule = 'hebbian'  # Learning rule
sync_target = 'sender'  # Target of the attack (sender or receiver)
num_simulations = 1000
sigma_comb_num = 2**(K-1)

def gen_sigma(K, tau):
    # Generate 2^(K-1) sigma combinations based on tau
    sigma_combinations = np.ones((sigma_comb_num, K), dtype=int)  # Initialize a matrix with all sigma values as 1

    for i in range(sigma_comb_num):
        bin_rep = np.array(list(bin(i)[2:].zfill(K-1)), dtype=int)  # Generate combinations in binary representation
        bin_rep = np.where(bin_rep == 1, -1, 1)  # Convert binary digits to 1 or -1
        sigma_combinations[i, :K-1] = bin_rep  # First K-1 elements

        # Set the last sigma based on tau to ensure the product of sigma equals tau
        sigma_combinations[i, K-1] = int(tau * np.prod(sigma_combinations[i, :K-1]))

    return sigma_combinations

# Pre-generate sigma combinations for tau = 1 and tau = -1
sigma_pos_tau = gen_sigma(K, 1)
sigma_neg_tau = gen_sigma(K, -1)

def attack_step(L, N, K, M, sync_target, rule):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)

    # Initialize the weights of the attacker swarm
    attacker_swarm_W = np.random.choice(np.arange(-L, L + 1), size=(1, K, N))

    target = sender if sync_target == 'sender' else receiver

    steps, sync_steps, attack_sync_steps = 0, None, None
    attack_success = False  # Record whether the attack is successful

    while True:
        steps += 1  # Record the number of steps

        # Generate random input vectors
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))

        # Update tau for sender and receiver, which recalculates sigma
        sender.update_tau(X)
        receiver.update_tau(X)

        # All networks in the attacker swarm should recalculate sigma and tau
        attacker_swarm_sigma = np.sign(
            np.sum(np.multiply(attacker_swarm_W, X), axis=2, keepdims=True)
        )
        # Replace 0 in sigma with -1
        attacker_swarm_sigma = np.where(attacker_swarm_sigma == 0, -1, attacker_swarm_sigma)
        attacker_swarm_tau = np.prod(attacker_swarm_sigma, axis=1, keepdims=True)

        # Synchronize attackers with sender or receiver
        if sender.tau == receiver.tau:
            # Update weights for sender and receiver
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)

            # Current number of attackers Q
            Q = attacker_swarm_W.shape[0]

            # If the number of attackers is less than M, expand attacker_swarm_W
            if Q < M:
                # Use np.repeat to repeat each K*N matrix sigma_comb_num times along the first dimension, generating (sigma_comb_num*Q, K, N)
                attacker_swarm_W_extend = np.repeat(attacker_swarm_W, repeats=sigma_comb_num, axis=0)

                # Each 4 attackers correspond to one sigma_combination
                sigma_combinations = sigma_pos_tau if target.tau == 1 else sigma_neg_tau

                # Update every sigma_comb_num matrices in attacker_swarm_W_extend
                for i in range(Q):
                    for j in range(sigma_comb_num):
                        idx = i * sigma_comb_num + j  # Group every sigma_comb_num
                        # Find the sigma that matches sender.tau and update the corresponding row of W
                        for row in range(K):
                            if sigma_combinations[j][row] == sender.tau:
                                # Update rule: only update rows that match sender.tau
                                attacker_swarm_W_extend[idx, row, :] += sender.tau * X[row, :]

                # After updating, replace the original attacker_swarm_W with the extended matrix
                attacker_swarm_W = attacker_swarm_W_extend
            
            else:
                # Filter attackers whose tau equals sender's tau
                matching_indices = np.where(attacker_swarm_tau.flatten() == sender.tau)[0]
                # Retain these matching attackers
                attacker_swarm_W = attacker_swarm_W[matching_indices, :, :]
                attacker_swarm_sigma = attacker_swarm_sigma[matching_indices, :, :]
                attacker_swarm_tau = attacker_swarm_tau[matching_indices]
                # Update the weights of the retained attackers
                for idx in range(attacker_swarm_W.shape[0]):
                    for row in range(K):
                        # Update weights using each attacker's own tau and sigma
                        if attacker_swarm_sigma[idx, row, :] == attacker_swarm_tau[idx]:
                            attacker_swarm_W[idx, row, :] += attacker_swarm_tau[idx].item() * X[row, :]

        # Limit the weight range to [-L, L]
        attacker_swarm_W = np.clip(attacker_swarm_W, -L, L)

        # Check if any attacker is synchronized
        for idx in range(attacker_swarm_W.shape[0]):
            # Compare each attacker's weight matrix with the sender's weight matrix
            if np.array_equal(attacker_swarm_W[idx], sender.W):
                attack_sync_steps = steps
                attack_success = True
                return attack_success, steps

        # Check if sender and receiver are synchronized
        if sender.is_sync(receiver, state='parallel'):
            sync_steps = steps
            break

    return attack_success, sync_steps

def run_simulation(L, N, K, M, sync_target, rule):
    success, steps = attack_step(L, N, K, M, sync_target, rule)
    return success

if __name__ == '__main__':
    success_count = 0

    # Use ProcessPoolExecutor for parallelization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, L, N, K, M, sync_target, rule) for _ in range(num_simulations)]
        
        for future in tqdm(as_completed(futures), total=num_simulations, desc=f'M={M}, N={N}'):
            if future.result():  # If the returned result is successful
                success_count += 1

    # Calculate and output the probability of a successful attack
    attack_success_probability = success_count / num_simulations
    print(f'Probability of successful attack: {100 * attack_success_probability:.2f}%')
