import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class TreeParityMachine:
    def __init__(self, L, N, K, zero_replace):
        self.L, self.N, self.K = L, N, K
        self.W = np.random.choice(np.arange(-L, L + 1), size=(K, N))
        self.zero_replace = zero_replace

    def update_tau(self, X):
        self.sigma = np.sign(np.sum(np.multiply(self.W, X), axis=1))
        self.sigma[self.sigma == 0] = self.zero_replace
        self.tau = np.prod(self.sigma)

    def update_W(self, X, rule='hebbian'):
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
        self.W = np.clip(self.W, -self.L, self.L)

    def is_sync(self, other, state='anti_parallel'):
        if state == 'anti_parallel':
            return np.array_equal(self.W, -other.W)
        elif state == 'parallel':
            return np.array_equal(self.W, other.W)

def geometric_attack(tpm1, tpm2, tpm3, rule='hebbian'):
    steps = 0
    while True:
        X = np.random.choice([-1, 1], size=(tpm1.K, tpm1.N))

        tpm1.update_tau(X)
        tpm2.update_tau(X)
        tpm3.update_tau(X)

        if tpm1.tau * tpm2.tau < 0:
            continue
        
        elif tpm1.tau == tpm2.tau == tpm3.tau:  # only when A, B and C output the sames
            tpm3.update_W(X, rule)
        
        else:  # when A and B output the same, but C is different
            # calculate the distance between the output of C and A, B
            min_distance = float('inf')
            min_index = -1
            for i in range(tpm1.K):
                distance = np.abs(np.dot(tpm3.W[i], X[i]))
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            
            # update the weights of C
            tpm3.W[min_index] -= X[min_index] * tpm3.sigma[min_index]
            tpm3.W = np.clip(tpm3.W, -tpm3.L, tpm3.L)
        
        steps += 1
        
        if tpm1.is_sync(tpm2) and tpm1.is_sync(tpm3):
            break
    
    return steps

def train_attack(L, N, K, num_runs=5000):
    tpm1 = TreeParityMachine(L, N, K, 1)
    tpm2 = TreeParityMachine(L, N, K, 1)
    tpm3 = TreeParityMachine(L, N, K, 1)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(geometric_attack, tpm1, tpm2, tpm3)
            for _ in range(num_runs)
        ]

        return [
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]

if __name__ == "__main__":
    L, N, K = 3, 100, 3
    attack_results = train_attack(L, N, K, num_runs=100)
    print(f"avg t_sync: {np.mean(attack_results)}")
