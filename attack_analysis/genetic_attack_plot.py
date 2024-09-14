import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))


data = pd.read_csv("./result/genetic_attack.csv")
unique_K = data['K'].unique()
unique_L = sorted(data['L'].unique())
cmap = plt.get_cmap('Set2')
colors = [cmap(i) for i in range(len(unique_L))]

plt.figure(figsize=(8, 6))

for idx, k_value in enumerate(unique_K):
    subset_K = data[data['K'] == k_value]
    
    for j, l_value in enumerate(unique_L):
        subset_L = subset_K[subset_K['L'] == l_value]
        
        if k_value == 2:
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker='s', markerfacecolor='none', markeredgecolor='black',
                     linestyle='--', color=colors[j], label=f'L = {l_value}' if idx == 0 else "")
            
            for x, y in zip(subset_L['M'], subset_L['Success Rate (%)']):
                plt.text(x, y+2.3, f'K=2\n{y:.2f}%', ha='center', color='black')
        else:
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker='o', markerfacecolor='none',
                     markeredgecolor=colors[j],
                     linestyle='--', color=colors[j], label=f'L = {l_value}' if idx == 0 else "")

plt.xticks(sorted(data['M'].unique()))
plt.xlabel('M')
plt.ylabel('Success Probability (%)')
plt.grid(True, linestyle='--', alpha=0.5)

legend_color = [plt.Line2D([0], [0], color=colors[j], linestyle='--') for j in range(len(unique_L))]
legend_labels = [f'L = {l_value}' for l_value in unique_L]
plt.legend(legend_color, legend_labels, loc='best')

plt.savefig("./figures/transparent/attack_genetic.png", transparent=True)

plt.show()
