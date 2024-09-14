import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("geometric_attack.csv")

# 假设 CSV 文件中有 "K", "N", "L", "Success Probability (%)" 四列
unique_K = data['K'].unique()
unique_N = data['N'].unique()
unique_L = sorted(data['L'].unique())  # 获取我们实际有的L值
markers = ['o', 's', '^']  # 用于区分不同 N 的点形状

# 使用 Set2 颜色映射
cmap = plt.get_cmap('Set2')
colors = [cmap(i) for i in range(len(unique_K))]  # 生成与 K 值数量相对应的颜色

plt.figure(figsize=(8, 6))

# 遍历不同的 K 值绘制不同的折线
for idx, k_value in enumerate(unique_K):
    subset_K = data[data['K'] == k_value]
    for i, n_value in enumerate(unique_N):
        subset = subset_K[subset_K['N'] == n_value]
        # 绘制折线，设置透明度，不透明的数据点
        plt.plot(subset['L'], subset['Success Probability (%)'], 
                 marker=markers[i % len(markers)], markerfacecolor='none',  # 空心点
                 markeredgecolor=colors[idx],  # 数据点边框颜色与K值匹配
                 linestyle='--', color=colors[idx])

# 只显示我们有的 L 作为横坐标
plt.xticks(unique_L)

plt.xlabel('L')
plt.ylabel('Success Probability (%)')
plt.grid(True, linestyle='--', alpha=0.5)

# 创建第一个图例 - K 对应的颜色
legend_K = [plt.Line2D([0], [0], color=colors[idx], linestyle='--') for idx in range(len(unique_K))]
legend_labels_K = [f'K = {k_value}' for k_value in unique_K]
legend1 = plt.legend(legend_K, legend_labels_K, loc=(0.84, 0.29))

# 创建第二个图例 - N 对应的形状
legend_N = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='black', markerfacecolor='none', linestyle='None') for i in range(len(unique_N))]
legend_labels_N = [f'N = {n_value}' for n_value in unique_N]
legend2 = plt.legend(legend_N, legend_labels_N, loc=(0.797, 0.68))

# 添加两个图例到图表
plt.gca().add_artist(legend1)  # 确保第一个图例不会被第二个覆盖

plt.show()
