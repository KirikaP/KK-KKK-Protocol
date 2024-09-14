import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM

def initialize_tpms(L, N, K):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)
    attacker = TPM(L, N, K, -1)

def probabilistic_attack(L, N, K, sync_target, rule):
    sender, receiver, attacker = initialize_tpms(L, N, K)
    target = sender if sync_target == 'sender' else receiver

    steps = 0

    while True:
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))
        sender.update_tau(X)
        receiver.update_tau(X)
        # attacker 怎么做

        if sender.tau == receiver.tau:
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
        
        # attacker 怎么做

        steps += 1
        
        # 检查攻击者是否同步
        if np.array_equal(attacker.W, sender.W):
            print(f'\nSuccess at {steps+1}')
            return 'success'
        
        # 检查 sender 和 receiver 是否同步
        if np.array_equal(sender.W, receiver.W):
            print(f'\nFail at {steps+1}')
            return 'failure'

def run_experiments(L, N, K, sync_target, rule, n_experiments):
    success_count = 0

    for _ in tqdm(range(n_experiments)):
        result = simple_attack(L, N, K, sync_target, rule)
        if result == 'success':
            success_count += 1

    success_rate = success_count / n_experiments
    return success_rate

# 参数设置
L = 3
N = 100
K = 3
sync_target = 'sender'  # 攻击者尝试同步 sender
rule = 'hebbian'  # 例如，更新权重的规则
n_experiments = 2000  # 运行 2000 次实验

# 运行实验并计算成功概率
success_rate = run_experiments(L, N, K, sync_target, rule, n_experiments)
print(f"攻击成功率: {success_rate * 100:.2f}%")
