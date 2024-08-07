import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


class KKNetwork:
    def __init__(self, L: int, N: int, K: int, zero_replacement: int):
        self.L = L
        self.L_list = np.arange(-L, L + 1)
        self.N = N
        self.K = K
        self.W = np.random.choice(self.L_list, size=(K, N))
        self.Y = np.zeros(K)
        self.O = 0
        self.zero_replacement = zero_replacement

    def get_Y(self, X: np.ndarray):
        self.Y = np.sign(np.sum(self.W * X, axis=1))
        self.Y[self.Y == 0] = self.zero_replacement

    def get_O(self):
        self.O = np.prod(self.Y)

    def update_weights(self, X: np.ndarray):
        for k in range(self.K):
            if self.O * self.Y[k] > 0:
                self.W[k] -= self.O * X
        self.W = np.clip(self.W, -self.L, self.L)


def single_update(L, N, K):
    S = KKNetwork(L, N, K, zero_replacement=1)
    R = KKNetwork(L, N, K, zero_replacement=-1)
    step_count = 0

    while True:
        X = np.random.choice([-1, 1], size=N)

        S.get_Y(X)
        S.get_O()

        R.get_Y(X)
        R.get_O()

        if S.O * R.O > 0:
            S.update_weights(X)
            R.update_weights(X)

        step_count += 1

        if np.array_equal(S.W, R.W):
            return step_count


def train_networks(L, N, K, num_runs=5000):
    step_counts = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(single_update, L, N, K) for _ in range(num_runs)]

        for future in tqdm(
            as_completed(futures),
            total=num_runs,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        ):
            result = future.result()
            if result is not None:
                step_counts.append(result)

    return step_counts


if __name__ == "__main__":
    L = 3
    K = 3
    N_values = [11, 21, 51, 101, 1001]
    avg_sync_times = []

    for N in N_values:
        step_counts = train_networks(L, N, K)
        avg_sync_time = np.mean(step_counts)
        avg_sync_times.append(avg_sync_time)

    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))

    for i, (N, color) in enumerate(zip(N_values, colors)):
        x = 1 / N
        y = avg_sync_times[i]
        plt.scatter(x, y, color=color, label=f'N={N}')
        plt.text(x, y, f'{y:.1f}', ha='left', va='bottom')

    plt.xlim([-0.005, max([1 / N for N in N_values]) + 0.012])
    plt.legend(loc='lower right')
    plt.xlabel('1/N')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time, L = {L}, K = {K}')
    plt.show()
