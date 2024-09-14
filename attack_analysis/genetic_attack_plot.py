import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("genetic_attack.csv")

# 假设 CSV 文件中有 "K", "N", "M", "L", "Success Rate (%)" 五列
unique_K = data['K'].unique()
unique_N = data['N'].unique()
unique_L = sorted(data['L'].unique())  # 获取我们实际有的L值
markers = ['o', 's', '^']  # 用于区分不同 N 的点形状

# 使用 Set2 颜色映射
cmap = plt.get_cmap('Set2')
colors = [cmap(i) for i in range(len(unique_L))]  # 生成与 L 值数量相对应的颜色

plt.figure(figsize=(8, 6))

# 遍历不同的 K 值绘制不同的折线
for idx, k_value in enumerate(unique_K):
    subset_K = data[data['K'] == k_value]
    for i, n_value in enumerate(unique_N):
        subset = subset_K[subset_K['N'] == n_value]
        # 遍历不同的 L 值绘制不同的颜色
        for j, l_value in enumerate(unique_L):
            subset_L = subset[subset['L'] == l_value]
            # 绘制折线，设置透明度，不透明的数据点
            plt.plot(subset_L['M'], subset_L['Success Rate (%)'], 
                     marker=markers[i % len(markers)], markerfacecolor='none',  # 空心点
                     markeredgecolor=colors[j],  # 数据点边框颜色与L值匹配
                     linestyle='--', color=colors[j], label=f'L = {l_value}' if i == 0 else "")

# 只显示我们有的 M 作为横坐标
plt.xticks(sorted(data['M'].unique()))

plt.xlabel('M (Number of Attackers)')
plt.ylabel('Success Rate (%)')
plt.grid(True, linestyle='--', alpha=0.5)

# 创建第一个图例 - L 对应的颜色
legend_L = [plt.Line2D([0], [0], color=colors[j], linestyle='--') for j in range(len(unique_L))]
legend_labels_L = [f'L = {l_value}' for l_value in unique_L]
legend1 = plt.legend(legend_L, legend_labels_L, loc='best')

# 添加两个图例到图表
plt.gca().add_artist(legend1)  # 确保第一个图例不会被第二个覆盖

plt.show()
