import matplotlib.pyplot as plt
from kk_multithread import train


if __name__ == "__main__":
    # Params
    L = 3
    K = 3
    N_values = [11, 101, 1001]
    colors = ['coral', 'green', 'black']
    labels = [f'N = {N}' for N in N_values]

    # Plot
    for N, color, label in zip(N_values, colors, labels):
        step_counts = train(L, N, K)
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

    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution, L = {L}, K = {K}')
    plt.legend()
    plt.show()
