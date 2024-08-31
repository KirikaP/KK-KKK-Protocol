import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from parity_machine import TreeParityMachine as TPM

# Parameters
L, K, N = 3, 3, 100
rule = 'anti_hebbian'
num_runs = 1

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Set a seed for reproducibility
    sender = TPM(L, N, K, zero_replace=1)
    receiver = TPM(L, N, K, zero_replace=-1)
    attacker = TPM(L, N, K, zero_replace=-1)

    step = 0
    sync_step = None

    while True:
        X = np.random.choice([-1, 1], size=(K, N))

        sender.update_tau(X)
        receiver.update_tau(X)
        attacker.update_tau(X)

        if sender.tau * receiver.tau < 0:
            sender.update_W(X, rule)
            receiver.update_W(X, rule)
            if sender.tau * attacker.tau < 0:
                attacker.update_W(X, rule)
        
        step += 1

        if sync_step is None:
            if sender.is_sync(receiver):
                sync_step = step

        if step % 50000 == 0:
            print(f"Sender and Receiver have synced at {sync_step}")
            matching_weights = np.sum(receiver.W == attacker.W)
            print(f"Step {step}: {matching_weights} out of {receiver.W.size} between attacker and receiver.")

        if np.array_equal(receiver.W, attacker.W):
            print(f"Attack successful at step {step}: All weights are synchronized.\n")
            break

    return step

if __name__ == "__main__":
    steps_to_sync = []

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, seed) for seed in range(num_runs)]
        
        for future in tqdm(as_completed(futures), total=num_runs):
            step = future.result()
            steps_to_sync.append(step)

    # Print the average steps to synchronization
    average_steps = np.mean(steps_to_sync)
    print(f"Average steps to synchronization: {average_steps}")
