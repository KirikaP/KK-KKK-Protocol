import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from parity_machine import TreeParityMachine as TPM


def initialize_attacker(sender, N, M):
    attacker = TPM(sender.L, N, sender.K, zero_replace=-1)
    for k in range(sender.K):
        # Randomly choose M indices to copy from sender
        indices_to_copy = np.random.choice(N, M, replace=False)
        for i in range(N):
            if i in indices_to_copy:
                attacker.W[k, i] = sender.W[k, i]  # Copy sender's weights
            else:
                # Randomly generate the remaining weights
                attacker.W[k, i] = np.random.choice(np.arange(-sender.L, sender.L + 1))
    return attacker


def attack_step(L, N, K, M=None, sync_target='sender', rule='hebbian', use_initialized_attacker=True):
    sender = TPM(L, N, K, zero_replace=-1)
    receiver = TPM(L, N, K, zero_replace=-1)
    
    if use_initialized_attacker and M is not None:
        attacker = initialize_attacker(sender, N, M)
    else:
        attacker = TPM(L, N, K, zero_replace=-1)

    sync_steps = None
    attack_sync_steps = None
    steps = 0
    target = sender if sync_target == 'sender' else receiver

    while True:
        steps += 1
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))

        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        if sender.tau * receiver.tau > 0:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
            attacker.update_W(X, rule=rule, tau_value=target.tau)

        if sync_steps is None and sender.is_sync(receiver, state='parallel'):
            sync_steps = steps
        if attack_sync_steps is None and attacker.is_sync(target, state='parallel'):
            attack_sync_steps = steps

        if sync_steps is not None and attack_sync_steps is not None:
            break

    if attack_sync_steps is not None:
        return sync_steps / attack_sync_steps
    else:
        return None


def run_simulations(L, N, K, M=None, sync_target='sender', rule='hebbian', num_simulations=1000, max_workers=8, use_initialized_attacker=True):
    ratios = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(attack_step, L, N, K, M, sync_target, rule, use_initialized_attacker)
            for _ in range(num_simulations)
        ]

        for future in tqdm(as_completed(futures), total=num_simulations):
            ratio = future.result()
            if ratio is not None:
                ratios.append(ratio)

    return ratios
