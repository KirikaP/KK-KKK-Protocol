import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./result/geometric_attack.csv")

unique_K = data['K'].unique()
unique_N = data['N'].unique()
unique_L = sorted(data['L'].unique())
markers = ['o', 's', '^']

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(unique_K))]
plt.figure()

for idx, k_value in enumerate(unique_K):
    subset_K = data[data['K'] == k_value]
    for i, n_value in enumerate(unique_N):
        subset = subset_K[subset_K['N'] == n_value]
        plt.plot(subset['L'], subset['Success Probability (%)'], 
                 marker=markers[i % len(markers)], markerfacecolor='none',
                 markeredgecolor=colors[idx],
                 linestyle='--', color=colors[idx])

plt.xticks(unique_L)
plt.yticks()
plt.xlabel('L')
plt.ylabel('Success Probability (%)')
plt.grid(True, linestyle='--', alpha=0.5)

legend_K = [plt.Line2D([0], [0], color=colors[idx], linestyle='--') for idx in range(len(unique_K))]
legend_labels_K = [f'K = {k_value}' for k_value in unique_K]
legend1 = plt.legend(legend_K, legend_labels_K, loc=(0.68, 0.73))

legend_N = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='black', markerfacecolor='none', linestyle='None') for i in range(len(unique_N))]
legend_labels_N = [f'N = {n_value}' for n_value in unique_N]
legend2 = plt.legend(legend_N, legend_labels_N, loc=(0.75, 0.305))

plt.gca().add_artist(legend1)
plt.tight_layout()

plt.savefig("./figures/transparent/geometric_attack.png", transparent=True)

plt.show()
