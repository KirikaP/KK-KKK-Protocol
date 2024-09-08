import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from scripts.parity_machine import TreeParityMachine as TPM


def initialize_attacker(sender, N, M):
    attacker = TPM(sender.L, N, sender.K, zero_replace=-1)
    for k in range(sender.K):
        # 随机选择 M 个索引，使其与 sender 的权重一致
        indices_to_copy = np.random.choice(N, M, replace=False)  # 随机选择 M 个索引
        
        for i in range(N):
            if i in indices_to_copy:
                attacker.W[k, i] = sender.W[k, i]  # 复制 sender 对应的权重
            else:
                # 其余的从 [-L, L] 中随机生成
                attacker.W[k, i] = np.random.choice(np.arange(-sender.L, sender.L + 1))
    
    return attacker

def attack_step(L, N, K, M, sync_target='sender', rule='hebbian'):
    sender = TPM(L, N, K, zero_replace=-1)
    receiver = TPM(L, N, K, zero_replace=-1)
    attacker = initialize_attacker(sender, N, M)  # Initialize attacker based on sender

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

        # check if sender is in sync with receiver
        if sync_steps is None and sender.is_sync(receiver, state='parallel'):
            sync_steps = steps

        # check if attacker is in sync with target
        if attack_sync_steps is None and attacker.is_sync(target, state='parallel'):
            attack_sync_steps = steps

        # if both sync_steps and attack_sync_steps are found, break
        if sync_steps is not None and attack_sync_steps is not None:
            break

    if attack_sync_steps is not None:
        return sync_steps / attack_sync_steps
    else:
        return None

def run_simulations(L, N, K, M, sync_target, rule, num_simulations=1000, max_workers=8):
    ratios = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(attack_step, L, N, K, M, sync_target, rule)
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
    rule = 'hebbian'
    sync_target = 'sender'
    num_simulations = 1000
    N = 100  # Now only run for N=100
    bin_width = 0.01
    colors = ['black', 'blue', 'red', 'green']

    # Define different values of M
    M_values = [N - 1, int(3/4 * N), int(1/2 * N), int(1/4 * N)]

    ratios_dict = {}

    for M in M_values:
        print(f"M = {M}")
        ratios = run_simulations(L, N, K, M, sync_target, rule, num_simulations=num_simulations, max_workers=16)
        ratios_dict[M] = ratios

    import matplotlib.pyplot as plt

    for idx, M in enumerate(M_values):
        ratios = ratios_dict[M]
        # Directly use ratios without filtering r < 1
        plt.hist(ratios, bins=np.arange(0, max(ratios) + 0.005, bin_width), edgecolor=colors[idx], label=f'M = {M}', histtype='step' if M not in [99, 25] else 'stepfilled')

    plt.xlabel('Ratio(r) (sync_steps / attack_sync_steps)')
    plt.ylabel('P(r)')
    plt.title(f'Distribution between Ratio(r) for Different M Values (N = {N})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="M Values")
    plt.show()
