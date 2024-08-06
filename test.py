import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

def initialize_weights(N, L):
    """Initialize weights for the network."""
    return np.random.randint(-L, L + 1, N)

def sign(x):
    """Sign function that returns 1 for non-negative and -1 for negative."""
    return 1 if x >= 0 else -1

def update_weights(weights, x, output, target, L):
    """Hebbian learning rule to update weights."""
    if output != target:
        for i in range(len(weights)):
            if output == sign(np.dot(weights, x)):
                weights[i] -= target * x[i]
                weights[i] = max(-L, min(L, weights[i]))  # Bound the weights

def run_simulation(N, L, max_steps=5000):
    """Run a single simulation of the network synchronization."""
    w_sender = initialize_weights(N, L)
    w_recipient = initialize_weights(N, L)

    t_sync = 0

    while t_sync < max_steps:
        x = np.random.choice([-1, 1], N)  # Random input
        y_sender = sign(np.dot(w_sender, x))
        y_recipient = sign(np.dot(w_recipient, x))

        output_sender = y_sender
        output_recipient = y_recipient

        if output_sender == output_recipient:
            break

        update_weights(w_sender, x, output_sender, output_recipient, L)
        update_weights(w_recipient, x, output_recipient, output_sender, L)

        t_sync += 1

    return t_sync

def plot_sync_times(N_values, L, num_trials=1000):
    """Plot the distribution of synchronization times for different network sizes."""
    for N in N_values:
        sync_times = []
        # Use tqdm to display the progress bar
        for _ in tqdm(range(num_trials), desc=f'Simulating for N={N}'):
            sync_time = run_simulation(N, L)
            sync_times.append(sync_time)
        
        plt.hist(sync_times, bins=40, alpha=0.6, label=f'N={N}')

    plt.xlabel('Number of Time Steps to Synchronization')
    plt.ylabel('Frequency')
    plt.title('Distribution of Synchronization Times')
    plt.legend()
    plt.show()

# Parameters
N_values = [11, 101, 1001]  # Different network sizes
L = 3  # Weight limit

plot_sync_times(N_values, L)
