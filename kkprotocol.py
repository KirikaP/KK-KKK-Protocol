import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

class Perception:
    def __init__(self, L: int, N: int):
        # L_list: set of {-L, -L+1, ..., 0, ..., L-1, L}
        #     2L+1 possible values in total
        # N: number of weights
        self.L = L
        self.L_list = np.arange(-L, L+1)
        self.N = N
        self.W = np.random.choice(self.L_list, size=self.N, replace=True)

def train_1_step(p1, p2, X: np.ndarray, L: int):
    # mutual learning, update the weights
    # p1/2: Perception objects
    # X: input data, shape (N, 1)
    # L: limit for the weights
    p1.sigma = np.sign(np.dot(p1.W.T, X))
    p2.sigma = np.sign(np.dot(p2.W.T, X))
    # Update the weights
    p1.W -= X.T * p1.sigma
    p2.W -= X.T * p2.sigma
    # Ensure p1.W and p2.W values are within the range [-L, L]
    p1.W = np.clip(p1.W, -L, L)
    p2.W = np.clip(p2.W, -L, L)

def run_simulation(L, N):
    A = Perception(L, N)
    B = Perception(L, N)
    start_time = time.time()

    while True:
        X = np.random.choice([-1, 1], size=N)
        if np.sign(np.dot(A.W.T, X)) == np.sign(np.dot(B.W.T, X)):
            train_1_step(A, B, X, L)
            if np.array_equal(A.W, B.W):
                break

    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

def main():
    L = 3
    num_runs = 1000
    N_values = [10, 40, 160, 640, 2560]
    average_times = []

    for N in N_values:
        # Use ProcessPoolExecutor to parallelize the runs
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_simulation, L, N) for _ in range(num_runs)]

            # Collect results as they are completed
            times = [future.result() for future in as_completed(futures)]

        average_time = np.mean(times)
        average_times.append(average_time)
        print(f"N = {N}: {num_runs} runs, Average time = {average_time:.4f} ms")

    # Plot the relationship between N and average run time
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, average_times, marker='o')
    plt.title('Relationship between N and Average Run Time')
    plt.xlabel('Number of Weights (N)')
    plt.ylabel('Average Run Time (ms)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
