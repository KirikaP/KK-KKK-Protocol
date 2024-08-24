import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class PM:
    """
    A class representing a KK network, which is a simplified model used to study synchronization
    between networks.

    Attributes:
    L : int
        The range limit for the weight values.
    N : int
        The number of columns in the weight matrix W.
    K : int
        The number of rows in the weight matrix W.
    zero_replace : int
        The value to replace 0 with in the output Y (1 for sender, -1 for receiver).
    W : numpy.ndarray
        The weight matrix of shape (K, N) with values ranging from -L to L.
    Y : numpy.ndarray
        The output vector after applying the update rule.
    O : int
        The overall output of the network, which is the product of the elements in Y.
    """
    def __init__(self, L, N, K, zero_replace):
        self.L, self.N, self.K = L, N, K
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace

    def update_O(self, X):
        """
        Update the output vector Y and overall output O based on the input matrix X.
        """
        self.Y = np.sign(np.sum(self.W * X, axis=1))
        self.Y[self.Y == 0] = self.zero_replace  # replace 0 with 1(sender) or -1(receiver)
        self.O = np.prod(self.Y)

    def update_W(self, X):
        """
        Update the weight matrix W based on the current output and input matrix X.
        """
        for k in range(self.K):
            if self.O * self.Y[k] > 0:  # if k-th unit is not sync
                self.W[k] -= self.O * X[k]  # updates W line by line
        self.W = np.clip(self.W, -self.L, self.L)  # clip W to [-L, L]

    def is_sync(self, other, case=-1):
        """
        Check if the current network is synchronized with another.
        """
        if case == -1:
            return np.array_equal(self.W, -other.W)
        elif case == 1:
            return np.array_equal(self.W, other.W)

    def sync(self, other):
        """
        Synchronize the current network with another network by iteratively updating weights.

        Parameters:
        other : PM
            Another instance of the PM network to synchronize with.
        stop_on_sync : bool, optional
            If True, stops the synchronization process once the networks are synchronized.
            Default is True.

        Returns:
        int
            The number of steps taken to achieve synchronization.
        """
        steps = 0
        while True:
            X = np.random.choice([-1, 1], size=(self.K, self.N))
            self.update_O(X)
            other.update_O(X)
            if self.O * other.O < 0:  # update rule according to paper
                self.update_W(X)
                other.update_W(X)
            steps += 1
            if self.is_sync(other):
                break

        return steps


def train_PMs(pm1, pm2, num_runs=5000):
    """
    Train two PM instances by running sync method in parallel.

    Parameters:
    pm1 (PM): The first PM instance.
    pm2 (PM): The second PM instance.
    num_runs (int): Number of parallel runs.
    stop_on_sync (bool): Whether to stop when the two networks synchronize.

    Returns:
    list: A list of sync steps for each run.
    """
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(pm1.sync, pm2)
            for _ in range(num_runs)
        ]

        return [
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]


# def train(L, N, K, num_runs=5000):
#     step_counts = []
#     for _ in range(num_runs):
#         sender = PM(L, N, K, 1)  # 创建第一个 PM 对象
#         receiver = PM(L, N, K, -1)  # 创建第二个 PM 对象
#         steps = sender.sync(receiver)  # 执行同步操作
#         step_counts.append(steps)  # 保存同步步数
#     return step_counts
