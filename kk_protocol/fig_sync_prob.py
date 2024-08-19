import matplotlib.pyplot as plt
import numpy as np
from scripts.kk_multithread import train, KKNetwork


def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    trials = len(sync_steps)
    steps = (
        range(1, max(sync_steps) + 1) if smooth else
        np.percentile(
            np.sort(sync_steps),
            np.linspace(0, 100, num_intervals + 1)
        )
    )
    return [
        (step, sum(1 for s in sync_steps if s <= step) / trials)
        for step in steps
    ]


if __name__ == "__main__":
    np.random.seed(114)
    L, K, N_values = 3, 3, [10, 16, 30, 1000]
    colors = ['red', 'green', 'blue', 'black']
    labels = [f'N = {N}' for N in N_values]
    markers = ['o', '^', 's', 'D']

    for N, color, label, marker in zip(N_values, colors, labels, markers):
        sync_steps = train(L, N, K, num_runs=5000)
        scatter_probs = calculate_probs(sync_steps)
        smooth_probs = calculate_probs(sync_steps, smooth=True)
        filtered_scatter_probs = [
            (step, prob) for step, prob in scatter_probs if prob >= 0.65
        ]

        if filtered_scatter_probs:
            plt.scatter(
                *zip(*filtered_scatter_probs),
                label=label,
                color=color,
                marker=marker,
                facecolors='none',
                edgecolors=color
            )

        for step, prob in scatter_probs:
            if prob == 1.0:
                plt.axvline(
                    x=step,
                    color=color,
                    linestyle=':',
                    alpha=0.6
                )  # 绘制虚线到x轴
                plt.text(
                    step,
                    1.02,
                    f'{step}',
                    ha='center',
                    va='bottom',
                    color=color
                )

        plt.plot(
            *zip(*smooth_probs),
            linestyle='--',
            color=color,
            alpha=0.6
        )

    plt.xlabel('Steps')
    plt.ylabel('Sync Prob')
    plt.title(f'Sync Prob vs. Steps (L = {L}, K = {K})')
    plt.legend()
    plt.grid(True)
    plt.show()
