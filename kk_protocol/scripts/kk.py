import numpy as np
import matplotlib.pyplot as plt


class KKNetwork:
    def __init__(self, L, N, K, zero_replace):
        self.L, self.N, self.K = L, N, K
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace

    def update_O(self, X):
        self.Y = np.sign(np.sum(self.W * X, axis=1))
        self.Y[self.Y == 0] = self.zero_replace  # replace 0 with 1(S) or -1(R)
        self.O = np.prod(self.Y)

    def update_W(self, X):
        for k in range(self.K):
            if self.O * self.Y[k] > 0:  # if k-th unit is not sync
                self.W[k] -= self.O * X[k]  # updates W line by line
        self.W = np.clip(self.W, -self.L, self.L)  # clip W to [-L, L]

    def is_sync(self, receiver):
        # check if the sender and receiver are sync
        return np.array_equal(self.W, -receiver.W)

    def sync(self, receiver, max_steps=10000):
        steps = 0
        while not self.is_sync(receiver) and steps < max_steps:
            X = np.random.choice([-1, 1], size=(self.K, self.N))
            self.update_O(X); receiver.update_O(X)
            if self.O * receiver.O < 0:  # update rule according to paper
                self.update_W(X)
                receiver.update_W(X)
            steps += 1
        return steps if self.is_sync(receiver) else None

def train(L, N, K, num_runs=5000):
    # 单线程执行多个同步任务
    step_counts = []
    for _ in range(num_runs):
        S = KKNetwork(L, N, K, 1)  # 创建第一个 KKNetwork 对象
        R = KKNetwork(L, N, K, -1)  # 创建第二个 KKNetwork 对象
        steps = S.sync(R)  # 执行同步操作
        step_counts.append(steps)  # 保存同步步数
    return step_counts


if __name__ == "__main__":
    L, K, N = 3, 3, 100
    step_counts = train(L, N, K)

    plt.hist(step_counts, bins=64, color='coral', histtype='barstacked')
    plt.xlabel('t_sync')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}, N = {N}')
    plt.grid(True)
    plt.show()
