import matplotlib.pyplot as plt
import numpy as np
from scripts.party_machine import PM, train_PMs


def filter_steps(L, K, N_values, max_t_sync=3000, num_runs=5000):
    N_step_counts = []
    for N in N_values:
        sender = PM(L, N, K, zero_replace=1)
        receiver = PM(L, N, K, zero_replace=-1)

        step_counts = train_PMs(sender, receiver, num_runs=num_runs)

        filtered_step_counts = [count for count in step_counts if count <= max_t_sync]
        N_step_counts.append(filtered_step_counts)

    return N_step_counts


if __name__ == "__main__":
    L, K, N_values = 3, 3, [10, 100, 1000]
    colors = ['green', 'orange', 'black']
    N_step_counts = filter_steps(L, K, N_values)

    for N, color, step_counts in zip(N_values, colors, N_step_counts):
        plt.hist(
            step_counts,
            bins=int(np.floor(max(step_counts)/40)),
            color=color,
            label=f'N = {N}',
            histtype='stepfilled' if N in [10, 100] else 'step',
            alpha=0.5 if N in [10, 100] else 1
        )

    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
