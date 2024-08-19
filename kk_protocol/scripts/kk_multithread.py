import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
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
    # 创建一个进程池执行器，以便并行执行多个任务
    with ProcessPoolExecutor() as executor:
        # 创建一个包含所有异步任务的列表，任务通过 executor.submit 提交
        futures = [
            # executor.submit 会提交一个任务到进程池中
            # 任务的内容是调用 KKNetwork(L, N, K, 1) 实例的 sync 方法
            # sync 方法的第一个参数 self 自动绑定到这个 KKNetwork 实例
            # sync 方法的第二个参数 other_network 则是另一个 KKNetwork 实例 KKNetwork(L, N, K, -1)
            # 换句话说，这里同时实例化了两个 KKNetwork 对象，并将它们用于同步操作
            executor.submit(
                KKNetwork(L, N, K, 1).sync,  # 第一个 KKNetwork 对象的 sync 方法
                KKNetwork(L, N, K, -1)  # 作为 sync 方法的参数传递的第二个 KKNetwork 对象
            ) for _ in range(num_runs)
        ]

        return [
            # as_completed 函数会在每个任务完成时生成一个 future 对象
            # future.result() 会阻塞直到任务完成，并返回任务的结果
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]


if __name__ == "__main__":
    L, K, N = 3, 3, 100
    step_counts = train(L, N, K)

    plt.hist(step_counts, bins=64, color='coral', histtype='barstacked')
    plt.xlabel('t_sync')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}, N = {N}')
    plt.grid(True)
    plt.show()
