import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from parity_machine import TreeParityMachine as TPM


def sync_with_bit_packages(tpm1, tpm2, B=10, rule='anti_hebbian', state='anti_parallel', stop_on_sync=False):
    steps = 0
    while True:
        # generate a batch of B input vectors
        X_batch = np.random.choice([-1, 1], size=(B, tpm1.K, tpm1.N))

        # to save the bit packages and corresponding sigma lists of both parties
        tpm1_bit_package = []
        tpm2_bit_package = []
        tpm1_sigma_package = []
        tpm2_sigma_package = []

        for X in X_batch:
            tpm1.update_tau(X)
            tpm2.update_tau(X)

            tpm1_bit_package.append(tpm1.tau)
            tpm2_bit_package.append(tpm2.tau)

            tpm1_sigma_package.append(tpm1.sigma.copy())
            tpm2_sigma_package.append(tpm2.sigma.copy())

        for i in range(B):
            steps += 1
            state_condition = (
                (tpm1_bit_package[i] * tpm2_bit_package[i] > 0) if state == 'parallel' 
                else (tpm1_bit_package[i] * tpm2_bit_package[i] < 0)
            )
            if state_condition:
                tpm1.update_W(X_batch[i], rule, tpm1_bit_package[i], tpm1_sigma_package[i])
                tpm2.update_W(X_batch[i], rule, tpm2_bit_package[i], tpm2_sigma_package[i])

        if tpm1.is_sync(tpm2, state):
            if stop_on_sync:
                return steps
            break

    return steps

def train_TPMs_with_bit_packages(tpm1, tpm2, B=10, num_runs=5000, rule='anti_hebbian', state='anti_parallel'):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(sync_with_bit_packages, tpm1, tpm2, B, rule, state)
            for _ in range(num_runs)
        ]

        results = []
        for future in tqdm(as_completed(futures), total=num_runs):
            result = future.result()
            results.append(result)
        return results


if __name__ == "__main__":
    # 测试比特包
    L, N, K = 3, 100, 3
    sender = TPM(L, N, K, 1)
    receiver = TPM(L, N, K, -1)

    B = 32  # 比特包大小
    num_runs = 5000  # 运行次数

    # 训练并输出结果
    results = train_TPMs_with_bit_packages(sender, receiver, B, num_runs, rule='anti_hebbian', state='anti_parallel')
    average_steps = np.mean(results)
    print(f"B={B}: {average_steps}")
