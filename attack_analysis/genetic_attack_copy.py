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
sigma_comb_num = 2**(K-1)

def gen_sigma(K, tau):
    # 根据 tau 生成 2^(K-1) 种 sigma 组合
    sigma_combinations = np.ones((2**(K-1), K), dtype=int)  # 初始化所有 sigma 都为 1 的矩阵

    for i in range(2**(K-1)):
        bin_rep = np.array(list(bin(i)[2:].zfill(K-1)), dtype=int)  # 二进制表示生成组合
        bin_rep = np.where(bin_rep == 1, -1, 1)  # 将二进制位转换为 1 或 -1
        sigma_combinations[i, :K-1] = bin_rep  # 前 K-1 个元素

        # 根据 tau 设定最后一个 sigma，确保 sigma 的乘积等于 tau
        sigma_combinations[i, K-1] = int(tau * np.prod(sigma_combinations[i, :K-1]))

    return sigma_combinations
# 预先生成 tau = 1 和 tau = -1 的 sigma 组合
sigma_pos_tau = gen_sigma(K, 1)
sigma_neg_tau = gen_sigma(K, -1)

def attack_step(L, N, K, M, sync_target, rule):
    sender = TPM(L, N, K, -1)
    receiver = TPM(L, N, K, -1)

    # 初始化攻击者群体的权重
    attacker_swarm_W = np.random.choice(np.arange(-L, L + 1), size=(1, K, N))

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

        # 攻击者群体的所有网络都应该重新计算 sigma 和 tau
        attacker_swarm_sigma = np.sign(
            np.sum(np.multiply(attacker_swarm_W, X), axis=2, keepdims=True)
        )
        # 将 sigma 中的 0 替换为 -1
        attacker_swarm_sigma = np.where(attacker_swarm_sigma == 0, -1, attacker_swarm_sigma)
        attacker_swarm_tau = np.prod(attacker_swarm_sigma, axis=1, keepdims=True)

        # 同步攻击者和发送者或接收者
        if sender.tau == receiver.tau:
            # 发送者和接收者更新权重
            sender.update_W(X, rule=rule)
            receiver.update_W(X, rule=rule)

            # 当前攻击者数量 Q
            Q = attacker_swarm_W.shape[0]

            # 当攻击者数量小于 M 时，扩展 attacker_swarm_W
            if Q < M:
                # 使用 np.repeat 沿第一个维度重复每个 K*N 矩阵 sigma_comb_num 次，生成 (sigma_comb_num*Q, K, N)
                attacker_swarm_W_extend = np.repeat(attacker_swarm_W, repeats=sigma_comb_num, axis=0)

                # 每 4 个攻击者对应一个 sigma_combination
                sigma_combinations = sigma_pos_tau if target.tau == 1 else sigma_neg_tau

                # 对 attacker_swarm_W_extend 的每 4 个矩阵进行更新
                for i in range(Q):
                    for j in range(4):
                        idx = i * 4 + j  # 每 4 个为一组
                        # 找到与 sender.tau 匹配的 sigma 并更新对应行的 W
                        for row in range(K):
                            if sigma_combinations[j][row] == sender.tau:
                                # 更新规则：只更新与 sender.tau 匹配的行
                                attacker_swarm_W_extend[idx, row, :] += sender.tau * X[row, :]

                # 更新完成后，用扩展后的矩阵替换原有的 attacker_swarm_W
                attacker_swarm_W = attacker_swarm_W_extend
            
            else:
                # 筛选出 tau 等于发送者 tau 的攻击者
                matching_indices = np.where(attacker_swarm_tau.flatten() == sender.tau)[0]
                # 保留这些匹配的攻击者
                attacker_swarm_W = attacker_swarm_W[matching_indices, :, :]
                attacker_swarm_sigma = attacker_swarm_sigma[matching_indices, :, :]
                attacker_swarm_tau = attacker_swarm_tau[matching_indices]
                # 更新保留下来的攻击者权重
                for idx in range(attacker_swarm_W.shape[0]):
                    for row in range(K):
                        # 使用每个攻击者自己的 tau 和 sigma 进行权重更新
                        if attacker_swarm_sigma[idx, row, :] == attacker_swarm_tau[idx]:
                            attacker_swarm_W[idx, row, :] += attacker_swarm_tau[idx].item() * X[row, :]

        # 限制权重范围 [-L, L]
        attacker_swarm_W = np.clip(attacker_swarm_W, -L, L)

        # 检查攻击者是否同步
        for idx in range(attacker_swarm_W.shape[0]):
            # 比较每个 attacker 的权重矩阵和 sender 的权重矩阵
            if np.array_equal(attacker_swarm_W[idx], sender.W):
                attack_sync_steps = steps
                attack_success = True
                print("\nsuccess")
                return attack_success, steps

        # 检查发送者和接收者是否同步
        if sender.is_sync(receiver, state='parallel'):
            sync_steps = steps
            print("\nfail")
            break

    return attack_success, sync_steps  # 返回攻击结果和同步的步数

def run_simulation(L, N, K, M, sync_target, rule):
    # 每个模拟任务运行的函数
    success, steps = attack_step(L, N, K, M, sync_target, rule)
    return success

if __name__ == '__main__':
    success_count = 0

    # 使用 ProcessPoolExecutor 进行并行化
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, L, N, K, M, sync_target, rule) for _ in range(num_simulations)]
        
        for future in tqdm(as_completed(futures), total=num_simulations, desc=f'Running M={M}, N={N}'):
            if future.result():  # 如果返回的结果是成功
                success_count += 1

    # 计算并输出攻击成功的概率
    attack_success_probability = success_count / num_simulations
    print(f'攻击成功的概率: {100 * attack_success_probability:.2f}%')
