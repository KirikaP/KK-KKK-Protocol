import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class TreeParityMachine:
    def __init__(self, L, N, K, zero_replace):
        self.L, self.N, self.K = L, N, K
        self.tau = None
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace

    def update_tau(self, X):
        self.sigma = np.sign(np.sum(np.multiply(self.W, X), axis=1))
        self.sigma[self.sigma == 0] = self.zero_replace  # replace 0 with 1(sender) or -1(receiver)
        self.tau = np.prod(self.sigma)

    def update_W(self, X, rule='anti_hebbian'):
        for k in range(self.K):
            if self.tau == self.sigma[k]:
                if rule == 'hebbian':
                    self.W[k] += self.sigma[k] * X[k]
                elif rule == 'anti_hebbian':
                    self.W[k] -= self.sigma[k] * X[k]
                elif rule == 'random_walk':
                    self.W[k] += X[k]
                else:
                    raise ValueError(f"Unknown learning rule: {rule}")

        self.W = np.clip(self.W, -self.L, self.L)  # clip W to [-L, L]

    def is_sync(self, other, state='anti_parallel'):
        if state == 'anti_parallel':
            return np.array_equal(self.W, -other.W)
        elif state == 'parallel':
            return np.array_equal(self.W, other.W)


def sync(tpm1, tpm2, rule='anti_hebbian'):
    steps = 0
    while True:
        X = np.random.choice([-1, 1], size=(tpm1.K, tpm1.N))
        tpm1.update_tau(X)
        tpm2.update_tau(X)

        if tpm1.tau * tpm2.tau < 0:
            tpm1.update_W(X, rule)
            tpm2.update_W(X, rule)
        steps += 1
        if tpm1.is_sync(tpm2):
            break

return steps


def train_TPMs(tpm1, tpm2, num_runs=5000, rule='anti_hebbian'):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(sync, tpm1, tpm2, rule)
            for _ in range(num_runs)
        ]

        return [
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]
