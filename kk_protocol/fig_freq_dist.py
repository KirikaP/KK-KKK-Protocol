import matplotlib.pyplot as plt
import numpy as np
from scripts.parity_machine import TreeParityMachine as TPM, train_TPMs


def filter_steps(L, K, N_values, max_t_sync=3000):
    N_step_counts = []
    for N in N_values:
        sender = TPM(L, N, K, 1)
        receiver = TPM(L, N, K, -1)

        step_counts = train_TPMs(sender, receiver)

        filtered_step_counts = [count for count in step_counts if count <= max_t_sync]
        N_step_counts.append(filtered_step_counts)

    return N_step_counts


if __name__ == "__main__":
    L, K, N_values = 3, 3, [10, 100, 1000]
    bin_width = 30
    colors = ['green', 'orange', 'black']
    N_step_counts = filter_steps(L, K, N_values)

    for N, color, step_counts in zip(N_values, colors, N_step_counts):
        bins = np.arange(0, 3000 + bin_width, bin_width)
        plt.hist(
            step_counts,
            bins=bins,
            color=color,
            label=f'N = {N}',
            histtype='stepfilled' if N in [10, 100] else 'step',
            alpha=0.5 if N in [10, 100] else 1
        )

    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}')
    plt.legend(loc='upper right')
    plt.xlim(0, 3000)
    plt.grid(True)
    plt.show()
