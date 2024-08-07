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
    L = 3
    K = 3
    N_values = [11, 101, 1001]
    colors = ['coral', 'green', 'black']
    labels = [f'N = {N}' for N in N_values]

    # Plot
    for N, color, label in zip(N_values, colors, labels):
        step_counts = train(L, N, K)
        histtype = 'stepfilled' if N in [11, 101] else 'step'
        alpha = 0.5 if N in [11, 101] else 1
        plt.hist(
            step_counts,
            bins=40,
            color=color,
            label=label,
            histtype=histtype,
            alpha=alpha
        )

    plt.xlabel('t_sync')
    plt.ylabel('P(t_sync)')
    plt.title(f'Distribution, L = {L}, K = {K}')
    plt.legend()
    plt.show()
