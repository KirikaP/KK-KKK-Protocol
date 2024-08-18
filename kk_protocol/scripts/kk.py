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
        self.zero_replacement = zero_replacement
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights for the network."""
        self.W = np.random.choice(self.L_list, size=(self.K, self.N))

    def update_O(self, X: np.ndarray):
        # X is now a K*N matrix where each row is the input for each hidden unit
        self.Y = np.sign(np.sum(self.W * X, axis=1))
        self.Y[self.Y == 0] = self.zero_replacement
        self.O = np.prod(self.Y)

    def update_weights(self, X: np.ndarray):
        for k in range(self.K):
            if self.O * self.Y[k] > 0:
                self.W[k] -= self.O * X[k]  # Update weights with the corresponding input vector

        # Apply boundary condition
        self.W = np.clip(self.W, -self.L, self.L)


def single_update(S, R):
    # Generate a unique K*N matrix for input X, where each row corresponds to a different hidden unit
    X = np.random.choice([-1, 1], size=(S.K, S.N))

    S.update_O(X)
    R.update_O(X)

    if S.O * R.O < 0:
        # print("S.O * R.O < 0")
        S.update_weights(X)
        R.update_weights(X)

    return np.array_equal(S.W, -R.W)


def train(L, N, K, zero_replacement, num_runs=5000):
    step_counts = []

    for _ in tqdm(range(num_runs)):
        # Create new networks for each run
        S = KKNetwork(L, N, K, zero_replacement=1)
        R = KKNetwork(L, N, K, zero_replacement=-1)

        steps = 0
        while True:
            # Perform a single update step
            if single_update(S, R):
                break
            steps += 1

        step_counts.append(steps)

    return step_counts


if __name__ == "__main__":
    # Parameters
    np.random.seed(114)
    L = 3
    K = 3
    N = 100

    # Train and get the convergence steps
    step_counts = train(L, N, K, zero_replacement=1, num_runs=5000)
    
    # Plotting the histogram
    plt.hist(
        step_counts,
        bins=64,
        color='coral',
        label='N = 100',
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
