import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class KKNetwork:
    def __init__(self, L: int, N: int, K: int):
        self.L = L
        self.L_list = np.arange(-L, L + 1)
        self.N = N
        self.K = K
        self.W = np.random.choice(self.L_list, size=(K, N))
        self.Y = np.zeros(K)
        self.O = 0

    def get_Y(self, X: np.ndarray):
        self.Y = np.sign(np.sum(self.W * X, axis=1))

    def get_O(self):
        self.O = np.prod(self.Y)

    def update_weights(self, X: np.ndarray):
        for k in range(self.K):
            if self.O * self.Y[k] > 0:
                self.W[k] -= self.O * X
        self.W = np.clip(self.W, -self.L, self.L)


def train(L, N, K, num_runs=5000):
    step_counts = []

    for _ in tqdm(range(num_runs)):
        S = KKNetwork(L, N, K)
        R = KKNetwork(L, N, K)
        step_count = 0

        while True:
            X = np.random.choice([-1, 1], size=(N))

            S.get_Y(X)
            S.Y[S.Y == 0] = 1
            S.get_O()

            R.get_Y(X)
            R.Y[R.Y == 0] = -1
            R.get_O()

            if S.O * R.O > 0:
                S.update_weights(X)
                R.update_weights(X)

            step_count += 1

            if np.array_equal(S.W, R.W):
                step_counts.append(step_count)
                break

    return step_counts
