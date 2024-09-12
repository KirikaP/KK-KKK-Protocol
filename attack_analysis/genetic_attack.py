import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM


# 参数设置
L = 3  # 权重范围
N = 100  # 输入大小
K = 3  # 感知器个数
M = 2500  # 最大攻击者数量
rule = 'hebbian'  # 学习规则
sync_target = 'sender'  # 攻击的目标（sender 或 receiver）
num_simulations = 1000

def initialize_tpms(L, N, K):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)
    attacker_swarm = [TPM(L, N, K, -1)]  # 初始攻击者
    return sender, receiver, attacker_swarm

def gen_sigma(K, tau):
    # 根据 tau 生成 2^(K-1) 种 sigma 组合
    sigma_combinations = []
    for i in range(2**(K-1)):
        sigma = [1] * K  # 默认所有 sigma 都是 1
        bin_rep = bin(i)[2:].zfill(K-1)  # 二进制表示，生成 2^(K-1) 种组合
        for j in range(K-1):
            sigma[j] = -1 if bin_rep[j] == '1' else 1
        # 根据 tau 设定最后一个 sigma，并确保转换为 int 类型
        sigma[K-1] = int(tau * np.prod(sigma[:K-1]))  # 保证 sigma 的乘积等于 tau
        sigma_combinations.append(sigma)
    return sigma_combinations
# 预先生成 tau = 1 和 tau = -1 的 sigma 组合
sigma_pos_tau = gen_sigma(K, 1)
sigma_neg_tau = gen_sigma(K, -1)

def attack_step(L, N, K, M, sync_target, rule):
    sender, receiver, attacker_swarm = initialize_tpms(L, N, K)
    target = sender if sync_target == 'sender' else receiver

    steps, sync_steps, attack_sync_steps = 0, None, None
    attack_success = False  # 记录攻击成功与否

    while True:
        steps += 1  # 记录步数

        # 生成随机输入向量
        X = np.random.choice([-1, 1], size=(sender.K, sender.N))

        # 更新发送者和接收者的 tau，这个过程会重新计算 sigma
        sender.update_tau(X)
        receiver.update_tau(X)

        # 攻击者群体的所有网络都应该重新计算 sigma
        for attacker in attacker_swarm:
            attacker.update_tau(X)  # 根据输入动态更新 tau 和 sigma

        # 同步攻击者和发送者或接收者
        if sender.tau == receiver.tau:
            # 发送者和接收者更新权重
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)
            # 生成攻击者变体，动态生成 sigma 组合，并生成新的攻击者
            if len(attacker_swarm) < M:
                sigma_combinations = sigma_pos_tau if target.tau == 1 else sigma_neg_tau
                new_attackers = []
                for attacker in attacker_swarm:
                    for sigma in sigma_combinations:
                        new_attacker = TPM(L, N, K, -1)
                        new_attacker.W = np.copy(attacker.W)  # 复制权重
                        new_attacker.update_W(X, rule=rule, tau_value=target.tau, sigma_value=sigma)
                        new_attackers.append(new_attacker)
                    # attacker.update_W(X, rule=rule, tau_value=target.tau)
                attacker_swarm.clear()
                attacker_swarm.extend(new_attackers)
            # 如果攻击者数量超过 M，执行筛选操作
            else:
                # 过滤攻击者群，只保留 tau 等于发送者 tau 的攻击者
                attacker_swarm = [attacker for attacker in attacker_swarm if attacker.tau == sender.tau]
                # 更新保留下来的攻击者网络的权重
                for attacker in attacker_swarm:
                    # attacker.update_W(X, rule=rule, tau_value=target.tau)
                    attacker.update_W(X, rule=rule)

        # 检查攻击者是否同步
        for attacker in attacker_swarm:
            if attacker.is_sync(target, state='parallel'):
                attack_sync_steps = steps
                attack_success = True
                print(" success")
                return attack_success, steps

        # 检查发送者和接收者是否同步
        if sender.is_sync(receiver, state='parallel'):
            sync_steps = steps
            print(" fail")
            break

    return attack_success, sync_steps  # 返回攻击结果和同步的步数

def run_simulation(L, N, K, M, sync_target, rule):
    # 每个模拟任务运行的函数
    success, steps = attack_step(L, N, K, M, sync_target, rule)
    return success

if __name__ == '__main__':
    success_count = 0
    num_workers = os.cpu_count()  # Automatically select the number of workers based on CPU cores

    # 使用 ProcessPoolExecutor 进行并行化
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_simulation, L, N, K, M, sync_target, rule) for _ in range(num_simulations)]
        
        for future in tqdm(as_completed(futures), total=num_simulations, desc=f'Running M={M}, N={N}'):
            if future.result():  # 如果返回的结果是成功
                success_count += 1

    # 计算并输出攻击成功的概率
    attack_success_probability = success_count / num_simulations
    print(f'攻击成功的概率: {100 * attack_success_probability:.2f}%')
