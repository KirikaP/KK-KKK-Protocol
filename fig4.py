import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Machine:
    def __init__(self, L: int, N: int, K: int):
        self.L = L
        self.L_list = np.arange(-L, L + 1)
        self.N = N
        self.K = K
        self.W = np.random.choice(self.L_list, size=(K, N), replace=True)
        self.sigma = np.zeros(K)
        self.tau = 0

    def update_tau(self, X: np.ndarray):
        self.sigma = np.sign(np.sum(self.W * X, axis=1))  # output of each perceptron
        self.tau = np.prod(self.sigma)  # total output tau

    def update_weights(self, X: np.ndarray):
        condition = self.sigma == self.tau
        for k in range(self.K):
            if condition[k]:
                self.W[k] -= X * self.sigma[k]
        self.W = np.clip(self.W, -self.L, self.L)


def train(L, N, K, num_runs=5000):
    step_counts = []

    for _ in tqdm(range(num_runs)):
        S = Machine(L, N, K)
        R = Machine(L, N, K)
        step_count = 0

        while True:
            X = np.random.choice([-1, 1], size=(N))
            S.update_tau(X)
            R.update_tau(X)

            if S.tau == R.tau:
                S.update_weights(X)
                R.update_weights(X)

            step_count += 1
            if np.array_equal(S.W, R.W):
                step_counts.append(step_count)
                break

    return step_counts

# Parameter configuration
L = 3
K = 3
N_values = [11, 101, 1001]
colors = ['r', 'g', 'b']
labels = [f'N = {N}' for N in N_values]

# Plot histograms
for N, color, label in zip(N_values, colors, labels):
    step_counts = train(L, N, K)
    plt.hist(step_counts, bins=40, color=color, edgecolor='black', linewidth=0.5, alpha=0.5, label=label)

plt.xlabel('t_sync')
plt.ylabel('P(t_sync)')
plt.title(f'Distribution of synchronization time for different N values, L = {L}, K = {K}')
plt.legend()
plt.show()
