import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KKNetwork:
    """
    L: integer weights bounded by [-L, L]
    K: number of hidden units
    N: each hidden unit size
    zero_replacement: value to replace zero with
    W: weights, a K*N matrix
    Y: hidden unit output, should be a list of K elements
    O: output bit, the product of all Y
    """
    def __init__(self, L: int, N: int, K: int, zero_replacement: int):
        self.L = L
        self.L_list = np.arange(-L, L + 1)
        self.N = N
        self.K = K
        self.W = np.random.choice(self.L_list, size=(K, N))
        self.Y = np.zeros(K)
        self.O = 0
        self.zero_replacement = zero_replacement

    def update_O(self, X: np.ndarray):
        # X is now a K*N matrix where each row is the input for each hidden unit
        self.Y = np.sign(np.sum(self.W * X, axis=1))
        self.Y[self.Y == 0] = self.zero_replacement
        self.O = np.prod(self.Y)

    def update_weights(self, X: np.ndarray):
        for k in range(self.K):
            if self.O * self.Y[k] > 0:
                self.W[k] -= self.O * X[k]  # Update weights with the corresponding input vector
        self.W = np.clip(self.W, -self.L, self.L)


def convergence_steps(S, R, max_steps=10000):
    step_count = 0

    while step_count < max_steps:
        # Generate a unique K*N matrix for input X, where each row corresponds to a different hidden unit
        X = np.random.choice([-1, 1], size=(S.K, S.N))

        S.update_O(X)
        R.update_O(X)

        if S.O * R.O > 0:
            S.update_weights(X)
            R.update_weights(X)

        step_count += 1

        if np.array_equal(S.W, R.W):
            break

    return step_count


def worker(S, R, max_steps=10000):
    S_copy = KKNetwork(S.L, S.N, S.K, S.zero_replacement)
    R_copy = KKNetwork(R.L, R.N, R.K, R.zero_replacement)
    S_copy.W = np.copy(S.W)
    R_copy.W = np.copy(R.W)
    return convergence_steps(S_copy, R_copy, max_steps=max_steps)


def train(S, R, num_runs=5000, max_steps=10000):
    step_counts = []

    for _ in tqdm(range(num_runs)):
        result = worker(S, R, max_steps=max_steps)
        step_counts.append(result)

    return step_counts


if __name__ == "__main__":
    # Parameters
    np.random.seed(114)
    L = 3
    K = 3
    N = 10  # Using N = 10 which is causing slow performance
    max_steps = 10000  # Add a maximum step count to avoid infinite loops

    # Plotting
    S = KKNetwork(L, N, K, zero_replacement=1)
    R = KKNetwork(L, N, K, zero_replacement=-1)
    step_counts = train(S, R, num_runs=100, max_steps=max_steps)  # Reduced num_runs for testing

    plt.hist(
        step_counts,
        bins=40,
        color='coral',
        label='N = 10',
        histtype='barstacked',
    )

    # Labels and Title
    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
