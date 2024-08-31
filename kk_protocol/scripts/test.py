import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import entropy


class TreeParityMachine:
    def __init__(self, K, N, L):
        self.K = K
        self.N = N
        self.L = L
        self.weights = np.random.randint(-L, L + 1, size=(K, N))

    def compute_tau(self, X):
        """Compute the overall output of the TreeParityMachine."""
        self.sigma = np.array([self._compute_sigma(X[i], self.weights[i]) for i in range(self.K)])
        self.tau = np.prod(self.sigma)
        return self.tau

    def is_synchronized(self, other_tpm):
        """Check if the weights are synchronized with another TreeParityMachine."""
        return np.array_equal(self.weights, other_tpm.weights)

    def _compute_sigma(self, x, w):
        """Private method to compute the output of a hidden unit."""
        h_i = np.dot(w, x) / np.sqrt(self.N)
        return sgn(h_i)  # Using the external sgn function


def sgn(x):
    """Sign function, returns -1 for x <= 0, 1 for x > 0."""
    return np.where(x > 0, 1, -1)


def get_L2_norm(tpm):
    """Compute the L2 norm of the weight matrix for a given TreeParityMachine."""
    return np.linalg.norm(tpm.weights)


def get_boundary_weight_ratio(tpm):
    """Compute the ratio of weights at the boundaries +L and -L for a given TreeParityMachine."""
    boundary_weights = np.sum((tpm.weights == tpm.L) | (tpm.weights == -tpm.L))
    return boundary_weights / tpm.weights.size


def calculate_entropy(tpm):
    """Compute the Shannon entropy of the weights for a given TreeParityMachine."""
    flat_weights = tpm.weights.flatten()
    value, counts = np.unique(flat_weights, return_counts=True)
    probabilities = counts / len(flat_weights)
    return entropy(probabilities)


class WeightUpdater:
    def __init__(self, tpm):
        self.tpm = tpm

    def update(self, X, tau, rule='hebbian'):
        """Public method to update the weights according to the specified rule."""
        if rule == 'hebbian':
            self._hebbian(X, tau)
        elif rule == 'anti_hebbian':
            self._anti_hebbian(X, tau)
        elif rule == 'random_walk':
            self._random_walk(X, tau)
        else:
            raise ValueError("Unknown learning rule")

    def _hebbian(self, X, tau):
        """Private method to update weights using the Hebbian learning rule."""
        for i in range(self.tpm.K):
            if self.tpm.sigma[i] == tau:
                self.tpm.weights[i] += X[i] * self.tpm.sigma[i]
                self.tpm.weights[i] = self._weight_clip(self.tpm.weights[i])

    def _anti_hebbian(self, X, tau):
        """Private method to update weights using the Anti-Hebbian learning rule."""
        for i in range(self.tpm.K):
            if self.tpm.sigma[i] == tau:
                self.tpm.weights[i] -= X[i] * self.tpm.sigma[i]
                self.tpm.weights[i] = self._weight_clip(self.tpm.weights[i])

    def _random_walk(self, X, tau):
        """Private method to update weights using the Random-Walk learning rule."""
        for i in range(self.tpm.K):
            if self.tpm.sigma[i] == tau:
                self.tpm.weights[i] += X[i]
                self.tpm.weights[i] = self._weight_clip(self.tpm.weights[i])

    def _weight_clip(self, w):
        """Private method to clip the weights within the range [-L, L]."""
        return np.clip(w, -self.tpm.L, self.tpm.L)


def run_simulation_once(K, N, L, rule):
    """Run a single simulation for the given parameters and learning rule."""
    tpm_A = TreeParityMachine(K, N, L)
    tpm_B = TreeParityMachine(K, N, L)
    updater_A = WeightUpdater(tpm_A)
    updater_B = WeightUpdater(tpm_B)

    while not tpm_A.is_synchronized(tpm_B):
        X = np.random.choice([-1, 1], size=(K, N))
        tau_A = tpm_A.compute_tau(X)
        tau_B = tpm_B.compute_tau(X)

        if tau_A == tau_B:
            updater_A.update(X, tau_A, rule)
            updater_B.update(X, tau_B, rule)

    length_sum = get_L2_norm(tpm_A)
    boundary_ratio = get_boundary_weight_ratio(tpm_A)
    entropy_value = calculate_entropy(tpm_A)
    
    return length_sum, boundary_ratio, entropy_value


if __name__ == '__main__':
    # Parameters
    N_values = [10**i for i in range(1, 5)]  # N = 10^1, 10^2, 10^3, 10^4
    K = 3
    L = 3  # Set a constant L value
    runs = 1000  # Number of simulations per configuration
    rules = ['hebbian', 'anti_hebbian', 'random_walk']
    
    # Initialize result containers
    results_length = {rule: [[] for _ in N_values] for rule in rules}
    results_boundary = {rule: [[] for _ in N_values] for rule in rules}
    results_entropy = {rule: [[] for _ in N_values] for rule in rules}

    # Run simulations for each N and rule using parallel processing
    with ProcessPoolExecutor() as executor:
        futures = []
        for N in N_values:
            for rule in rules:
                for _ in range(runs):  # Submit each run individually
                    futures.append(executor.submit(run_simulation_once, K, N, L, rule))

        # Using tqdm for the progress bar
        progress_bar = tqdm(as_completed(futures), total=len(futures))
        for future in progress_bar:
            length_result, boundary_result, entropy_result = future.result()
            index = futures.index(future)
            rule = rules[(index // runs) % len(rules)]
            N_idx = index // (runs * len(rules))
            N = N_values[N_idx]
            progress_bar.set_description(f"Rule={rule}, N={N}")
            progress_bar.refresh()  # Manually refresh the progress bar
            
            # Append the result to the corresponding N and rule
            results_length[rule][N_idx].append(length_result)
            results_boundary[rule][N_idx].append(boundary_result)
            results_entropy[rule][N_idx].append(entropy_result)

    # Averaging the results for each (N, rule) combination
    for rule in rules:
        results_length[rule] = [np.mean(values) for values in results_length[rule]]
        results_boundary[rule] = [np.mean(values) for values in results_boundary[rule]]
        results_entropy[rule] = [np.mean(values) for values in results_entropy[rule]]

    # Plotting the results on the same canvas with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # First subplot: Length of weight vectors
    for rule, marker in zip(rules, ['o', 's', 'd']):
        ax1.plot(N_values, results_length[rule], marker=marker, linestyle='--', label=f'{rule.capitalize()} learning')
    ax1.set_xlabel('N (log scale)')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$L2\ Norm$')
    ax1.set_title('Length of the weight vectors in the steady state')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Proportion of weights at +/- L
    for rule, marker in zip(rules, ['o', 's', 'd']):
        ax2.plot(N_values, results_boundary[rule], marker=marker, linestyle='--', label=f'{rule.capitalize()} learning')
    ax2.set_xlabel('N (log scale)')
    ax2.set_xscale('log')
    ax2.set_ylabel('Proportion of Weights at +/- L')
    ax2.set_title('Proportion of Weights at Boundaries (+/-L) in the Steady State')
    ax2.legend()
    ax2.grid(True)

    # Third subplot: Shannon Entropy of weight vectors
    for rule, marker in zip(rules, ['o', 's', 'd']):
        ax3.plot(N_values, results_entropy[rule], marker=marker, linestyle='--', label=f'{rule.capitalize()} learning')
    ax3.set_xlabel('N (log scale)')
    ax3.set_xscale('log')
    ax3.set_ylabel('Shannon Entropy')
    ax3.set_title('Shannon Entropy of the weight vectors in the steady state')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
