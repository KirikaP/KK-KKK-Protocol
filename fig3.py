import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kk_multithread import KKNetwork, train, single_update


if __name__ == "__main__":
    """
    在 Windows 上, multiprocessing 模块通过导入主模块来创建新进程
    如果模块级代码没有用 if __name__ == "__main__": 保护
    那么每次创建新进程时都会执行模块的顶层代码
    1. 这可能导致无限循环，每个进程创建更多进程，最终导致 BrokenProcessPool 或耗尽系统资源
    2. 这个保护确保了像 ProcessPoolExecutor 这样的代码只在主进程中执行
       这防止了意外的副作用，并确保每个新进程中仅运行必要的代码部分
    """
    # Params
    np.random.seed(1337)
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

    # Plot with different colors for each point
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    for i, (N, color) in enumerate(zip(N_values, colors)):
        x = 1 / N
        y = avg_sync_times[i]
        plt.scatter(x, y, color=color, label=f'N={N}')
        plt.text(x, y, f'{y:.1f}', ha='left', va='bottom')

    # Manually set the x-axis range
    plt.xlim([-0.005, max([1/N for N in N_values]) + 0.012])

    # Add legend in the lower right corner
    plt.legend(loc='lower right')

    plt.xlabel('1/N')
    plt.ylabel('Average t_sync')
    plt.title(f'Average Synchronization Time, L = {L}, K = {K}')
    plt.show()
