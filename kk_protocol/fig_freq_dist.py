import matplotlib.pyplot as plt
import numpy as np
from scripts.kk_multithread import train, KKNetwork


if __name__ == "__main__":
    # Parameters
    L = 3
    K = 3
    N_values = [11, 101, 1001]
    colors = ['coral', 'green', 'black']
    labels = [f'N = {N}' for N in N_values]

    # Plotting
    for N, color, label in zip(N_values, colors, labels):
        S = KKNetwork(L, N, K, zero_replacement=1)
        R = KKNetwork(L, N, K, zero_replacement=-1)
        step_counts = train(S, R)
        histtype = 'stepfilled' if N in [11, 101] else 'step'
        alpha = 0.5 if N in [11, 101] else 1
        plt.hist(
            step_counts,
            bins=40,
            color=color,
            label=label,
            histtype=histtype,
            alpha=alpha
        )

    # Labels and Title
    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution of t_sync, L = {L}, K = {K}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
