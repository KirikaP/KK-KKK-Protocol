import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_simple_attack(csv_file, bin_width=0.01):
    # 读取 CSV 文件
    data = pd.read_csv(csv_file)
    
    # 按照 N 值进行分组
    N_values = sorted(data['N'].unique())
    
    for N in N_values:
        subset = data[data['N'] == N]
        truncated_ratios = subset[subset['Ratio'] < 1.0]['Ratio']

        # 提取成功率
        success_rate = subset['Success Rate'].iloc[0]

        # 绘制直方图，设置黑色边框
        plt.hist(
            truncated_ratios,
            bins=np.arange(0, 1 + 0.005, bin_width),
            alpha=0.5, label=f'N = {N} ({success_rate:.2%})',
            edgecolor='black', histtype='stepfilled'
        )

    plt.xlabel('Ratio(r) (sync_steps / attack_sync_steps)')
    plt.ylabel('P(r)')
    plt.title('Distribution between Ratio(r) and N Values')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="N Values")
    plt.show()

if __name__ == "__main__":
    csv_file = "simple_attack.csv"
    plot_simple_attack(csv_file)
