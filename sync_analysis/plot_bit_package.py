import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def load_results_from_custom_csv(file_path):
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    df.columns = [(rule, int(N)) for rule, N in df.columns]
    all_results = {}
    for (rule, N) in df.columns:
        if rule not in all_results:
            all_results[rule] = {}
        all_results[rule][N] = df[(rule, N)].values
    return df.index.values, all_results

if __name__ == "__main__":
    csv_file_path = "./result/bit_package_multi_rule.csv"
    B_values, all_results = load_results_from_custom_csv(csv_file_path)
    N_values = list(set(int(N) for rule in all_results for N in all_results[rule]))
    rules = list(all_results.keys())

    tab10_colors = plt.get_cmap('tab10')
    rule_colors = {
        'hebbian': tab10_colors(0),
        'anti_hebbian': tab10_colors(1),
        'random_walk': tab10_colors(2)
    }

    available_markers = ['o', 's', '^']
    markers = {N: available_markers[i] for i, N in enumerate(N_values)}

    all_y_values = np.concatenate([all_results[rule][N] for rule in rules for N in N_values])

    y_min_data = min(all_y_values) - 50
    y_max_data = max(all_y_values) + 50
    y_min = 100 * (np.floor(y_min_data / 100))
    y_max = 100 * (np.ceil(y_max_data / 100))
    y_ticks = np.arange(y_min, y_max + 100, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, rule in enumerate(rules):
        for N in N_values:
            ax.plot(
                np.log2(B_values), 
                all_results[rule][N], 
                zs=i,  # 将学习规则映射到y轴
                zdir='y', 
                marker=markers[N], 
                markerfacecolor='none', 
                linestyle='--', 
                color=rule_colors[rule]
            )

    ax.set_xlabel(r"log$_2{b}$")
    ax.set_ylabel("Learning Rule")
    ax.set_zlabel("Average Synchronization Steps")
    
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([rule.capitalize() for rule in rules])

    ax.set_zlim(y_min_data, y_max_data)
    ax.set_zticks(y_ticks)  # 设置Z轴刻度，每100显示一个

    plt.tight_layout()
    plt.savefig("./figures/transparent/t_sync_bit_package_multi_rule_loaded_3d.png", transparent=True)
    plt.show()
