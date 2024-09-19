import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from train import train_TPMs

def simulate(
    L, K, N, rule, num_runs=5000, zero_replace_1=-1, zero_replace_2=-1, state='parallel', B=None
):
    if B is not None:
        print(f"Running N = {N} with rule = {rule}, B = {B}")
    else:
        print(f"Running N = {N} with rule = {rule}")
    sync_steps = train_TPMs(L, K, N, zero_replace_1, zero_replace_2, num_runs, rule, state, B=B)
    return sync_steps

def calculate_probs(sync_steps, num_intervals=20, smooth=False):
    trials = len(sync_steps)
    if smooth:
        steps = np.linspace(1, max(sync_steps), num_intervals)  # Using linspace for fractional steps
    else:
        steps = np.percentile(np.sort(sync_steps), np.linspace(0, 100, num_intervals + 1))

    return [(step, sum(1 for s in sync_steps if s <= step) / trials) for step in steps]

def plot_sync_probs(simulation_results, L, K, num_intervals=20, smooth=True):
    colors = ['red', 'green', 'blue', 'purple']
    markers = ['o', '^', 's', 'd']
    labels = list(simulation_results.keys())

    for label, color, marker in zip(labels, colors, markers):
        sync_steps = simulation_results[label]
        scatter_probs = calculate_probs(sync_steps, num_intervals, smooth=False)
        smooth_probs = calculate_probs(sync_steps, num_intervals, smooth=smooth)
        
        plt.scatter(
            *zip(*scatter_probs),
            label=f'{label}', marker=marker, facecolors='none', edgecolors=color
        )

        plt.plot(*zip(*smooth_probs), linestyle='--', color=color, alpha=0.6)

        for step, prob in scatter_probs:
            if prob == 1.0:
                plt.text(int(step), 1.02, f'{int(step)}', ha='center', va='bottom', color=color)

    plt.xlabel('Steps')
    plt.ylabel('Sync Probability')
    y_ticks = np.linspace(0, 1, 11)
    plt.yticks(y_ticks)
    plt.ylim(0, 1.1)  # 将y轴最大值设置为1.1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./figures/transparent/t_sync_probability.png', transparent=True)
    plt.show()

def save_results_to_csv(simulation_results, file_path):
    df = pd.DataFrame(dict([(label, pd.Series(sync_steps)) for label, sync_steps in simulation_results.items()]))    
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    L, K = 3, 3
    N = 100
    num_runs = 5000
    rules = ['hebbian', 'anti_hebbian', 'random_walk']
    bit_package = 8

    simulation_results = {}

    for rule in rules:
        simulation_results[rule] = simulate(L, K, N, rule, num_runs)

    # 运行 anti_hebbian 规则，B=8 的情况
    simulation_results[f'anti_hebbian_B={bit_package}'] = simulate(L, K, N, 'anti_hebbian', num_runs, B=bit_package)

    plot_sync_probs(simulation_results, L, K)

    save_results_to_csv(simulation_results, './result/sync_results_N100_with_anti_hebbian_B8.csv')
