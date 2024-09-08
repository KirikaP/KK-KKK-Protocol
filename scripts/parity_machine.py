import numpy as np


class TreeParityMachine:
    """
    TreeParityMachine class for simulating the TPM synchronization process

    Attributes:
        L (int): Weight limit range [-L, L]
        N (int): Number of input bits per hidden unit
        K (int): Number of hidden units
        W (np.ndarray): Weight matrix of shape (K, N)
        zero_replace (int): Value to replace zero in sigma
    """
    def __init__(self, L, N, K, zero_replace):
        self.L = L
        self.N = N
        self.K = K
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace
        self.sigma = None
        self.tau = None

    def update_tau(self, X):
        """
        Update the tau value based on the input vector X and current weights
        
        Args:
            X (np.ndarray): Input vector of shape (K, N)
        """
        self.sigma = np.sign(np.sum(np.multiply(self.W, X), axis=1))
        self.sigma = np.where(self.sigma == 0, self.zero_replace, self.sigma)
        self.tau = np.prod(self.sigma)

    def update_W(self, X, rule='anti_hebbian', tau_value=None, sigma_value=None):
        """
        Update the weights based on the provided learning rule and input vector

        Args:
            X (np.ndarray): Input vector of shape (K, N)
            rule (str): Learning rule to apply ('hebbian', 'anti_hebbian', 'random_walk')
            tau_value (int): External tau value (default is self.tau)
            sigma_value (np.ndarray): External sigma values (default is self.sigma)
        
        Raises:
            ValueError: If an invalid rule is provided
        """
        # Use provided or default tau and sigma values
        tau = tau_value if tau_value is not None else self.tau
        sigma = sigma_value if sigma_value is not None else self.sigma

        # Rule map to simplify weight updates
        rule_map = {
            'hebbian': lambda W, tau, X: W + tau * X,
            'anti_hebbian': lambda W, tau, X: W - tau * X,
            'random_walk': lambda W, tau, X: W + X
        }
        if rule not in rule_map:
            raise ValueError("Invalid rule value. Choose 'hebbian', 'anti_hebbian', or 'random_walk'")

        for k in range(self.K):
            if tau == sigma[k]:
                self.W[k] = rule_map[rule](self.W[k], tau, X[k])

        self.W = np.clip(self.W, -self.L, self.L)

    def is_sync(self, other, state='anti_parallel'):
        """
        Check if the weights are synchronized with another TPM

        Args:
            other (TreeParityMachine): Another TPM instance
            state (str): Synchronization state to check ('parallel' or 'anti_parallel')

        Returns:
            bool: True if synchronized, False otherwise

        Raises:
            ValueError: If an invalid state is provided
        """
        if state == 'anti_parallel':
            return np.array_equal(self.W, -other.W)
        elif state == 'parallel':
            return np.array_equal(self.W, other.W)
        else:
            raise ValueError("Invalid state value. Choose 'parallel' or 'anti_parallel'")
