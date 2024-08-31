import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class TreeParityMachine:
    def __init__(self, L, N, K, zero_replace):
        self.L, self.N, self.K = L, N, K
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace

    def update_tau(self, X):
        self.sigma = np.sign(np.sum(np.multiply(self.W, X), axis=1))
        self.sigma[self.sigma == 0] = self.zero_replace  # replace 0 with 1(sender) or -1(receiver)
        self.tau = np.prod(self.sigma)

    def update_W(self, X):
        for k in range(self.K):
            if self.tau * self.sigma[k] > 0:  # if k-th unit is not sync
                self.W[k] -= self.sigma[k] * X[k]  # updates W line by line
        self.W = np.clip(self.W, -self.L, self.L)  # clip W to [-L, L]

    # def is_sync(self, other, case=-1):
    def is_sync(self, other, case=-1):
        if case == -1:
            return np.array_equal(self.W, -other.W)
        elif case == 1:
            return np.array_equal(self.W, other.W)

    def sync(self, other):
        steps = 0
        while True:
            X = np.random.choice([-1, 1], size=(self.K, self.N))
            self.update_tau(X)
            other.update_tau(X)
            if self.tau * other.tau < 0:  # update rule according to paper
            # if self.tau * other.tau > 0:
            # if self.tau == other.tau:
                self.update_W(X)
                other.update_W(X)
            steps += 1
            if self.is_sync(other):
                break

        return steps


def train_TPMs(tpm1, tpm2, num_runs=5000):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(tpm1.sync, tpm2)
            for _ in range(num_runs)
        ]

        return [
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]

