import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kk import KKNetwork, train

# Params
np.random.seed(0)
L = 3
K = 3
N_values = [11, 101, 1001]
colors = ['lightred', 'lightgreen', 'lightblue']
labels = [f'N = {N}' for N in N_values]

# Plot
for N, color, label in zip(N_values, colors, labels):
    step_counts = train(L, N, K)
    plt.hist(
        step_counts,
        bins=40,
        color=color,
        label=label,
        histtype='step'
    )

plt.xlabel('t_sync')
plt.ylabel('P(t_sync)')
plt.title(f'Distribution, L = {L}, K = {K}')
plt.legend()
plt.show()
