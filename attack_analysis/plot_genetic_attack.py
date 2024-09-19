import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

# Load the data
data = pd.read_csv("./result/genetic_attack.csv")
unique_K = data['K'].unique()
unique_L = sorted(data['L'].unique())

# Create color map
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(unique_L))]

plt.figure()

# Iterate over K values
for idx, k_value in enumerate(unique_K):
    subset_K = data[data['K'] == k_value]
    
    # Iterate over L values
    for j, l_value in enumerate(unique_L):
        subset_L = subset_K[subset_K['L'] == l_value]
        
        if k_value == 2 and l_value == 5:
            # Special point for K=2, L=5 (square marker)
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker='s', markerfacecolor='none', markeredgecolor='black',
                     label='K=2, L=5 (Square)')
            
        elif k_value == 5 and l_value == 2:
            # Special point for K=5, L=2 (triangle marker)
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker='^', markerfacecolor='none', markeredgecolor='black',
                     label='K=5, L=2 (Triangle)')
            
        else:
            # Default plot for other K and L values
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker='o', markerfacecolor='none',
                     markeredgecolor=colors[j],
                     linestyle='--', color=colors[j], label=f'L = {l_value}' if idx == 0 else "")

# Set axis labels and ticks with larger font size
plt.xticks(sorted(data['M'].unique()))
plt.yticks()
plt.xlabel('M')
plt.ylabel('Success Probability (%)')
plt.grid(True, linestyle='--', alpha=0.5)

# Create the custom legend for L values and special markers
legend_color = [plt.Line2D([0], [0], color=colors[j], linestyle='--') for j in range(len(unique_L))]
legend_labels = [f'L = {l_value}' for l_value in unique_L]

# Add special markers for K=2, L=5 and K=5, L=2
legend_color.append(plt.Line2D([0], [0], marker='s', color='black', label='K=2, L=5 (Square)', linestyle='None', markerfacecolor='none'))
legend_color.append(plt.Line2D([0], [0], marker='^', color='black', label='K=5, L=2 (Triangle)', linestyle='None', markerfacecolor='none'))
legend_labels.extend(['K=2, L=5', 'K=5, L=2'])

# Show the legend with a larger font size
plt.legend(legend_color, legend_labels, loc=(0.02, 0.23))
plt.tight_layout()

# Save the plot
plt.savefig("./figures/transparent/attack_genetic.png", transparent=True)

# Show the plot
plt.show()
