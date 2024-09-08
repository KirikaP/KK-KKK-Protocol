import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from scripts.parity_machine import TreeParityMachine as TPM


def attack_step(L, N, K, sync_target='sender', rule='anti_hebbian'):
    sender = TPM(L, N, K, zero_replace=-1)
    receiver = TPM(L, N, K, zero_replace=-1)
    attacker = TPM(L, N, K, zero_replace=-1)

    sync_steps = None
    attack_sync_steps = None
    steps = 0

    target = sender if sync_target == 'sender' else receiver

    while True:
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))

        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        if sender.tau * receiver.tau > 0:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
            attacker.update_W(X, rule=rule, tau_value=target.tau)

        # check if sender is in sync with receiver
        if sync_steps is None and sender.is_sync(receiver, state='parallel'):
            sync_steps = steps

        # check if attacker is in sync with target
        if attack_sync_steps is None and attacker.is_sync(target, state='parallel'):
            attack_sync_steps = steps

        # if both sync_steps and attack_sync_steps are found, break
        if sync_steps is not None and attack_sync_steps is not None:
            break

        steps += 1

    if attack_sync_steps is not None:
        return sync_steps / attack_sync_steps
    else:
        return None

def attacker_learn(L, N, K, sync_target, rule, num_simulations=1000, max_workers=8):
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


if __name__ == "__main__":
    # Parameters
    L, K = 3, 3
    rule = 'random_walk'
    sync_target = 'sender'
    num_simulations = 1000
    N_values = [100, 50, 20, 10]
    bin_width = 0.01

    ratios_dict = {}

    for N in N_values:
        print(f"N = {N}")
        ratios = attacker_learn(L, N, K, sync_target, rule, num_simulations=num_simulations, max_workers=8)  # 增加max_workers
        ratios_dict[N] = ratios

    import matplotlib.pyplot as plt

    for N, ratios in ratios_dict.items():
        # only keep ratios < 1.0
        truncated_ratios = [r for r in ratios if r < 1.0]

        if N == 100:
            plt.hist(truncated_ratios, bins=np.arange(0, 1 + 0.005, bin_width), edgecolor='black', label=f'N = {N}', histtype='step')
        elif N == 10:
            plt.hist(truncated_ratios, bins=np.arange(0, 1 + 0.005, bin_width), edgecolor='blue', label=f'N = {N}', histtype='step')
        else:
            plt.hist(truncated_ratios, bins=np.arange(0, 1 + 0.005, bin_width), alpha=0.5, label=f'N = {N}')

    plt.xlabel('Ratio(r) (sync_steps / attack_sync_steps)')
    plt.ylabel('P(r)')
    plt.title(f'Distribution between Ratio(r) and N Values')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="N Values")
    plt.show()
