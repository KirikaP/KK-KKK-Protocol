import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kk import KKNetwork, train

# Params
np.random.seed(0)
L = 3
K = 3
N_values = [11, 21, 51, 101, 1001]
labels = [f'N = {N}' for N in N_values]

# Calculate average sync time for each N
avg_sync_times = []
for N in N_values:
    step_counts = train(L, N, K)
    avg_sync_time = np.mean(step_counts)
    avg_sync_times.append(avg_sync_time)

# Plot
plt.plot(1/np.array(N_values), avg_sync_times, marker='o')
plt.xlabel('1/N')
plt.ylabel('Average t_sync')
plt.title(f'Average Synchronization Time, L = {L}, K = {K}')
plt.grid(True)
plt.show()
