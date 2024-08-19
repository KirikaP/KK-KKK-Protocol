import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .party_machine import KKNetwork  # 从 party_machine.py 中导入 KKNetwork 类，"." 表示当前目录
import matplotlib.pyplot as plt


def train_pm(L, N, K, num_runs=5000):
    # 创建一个进程池执行器，以便并行执行多个任务
    with ProcessPoolExecutor() as executor:
        # 创建一个包含所有异步任务的列表，任务通过 executor.submit 提交
        futures = [
            # executor.submit 会提交一个任务到进程池中
            # 任务的内容是调用 KKNetwork(L, N, K, 1) 实例的 sync 方法
            # sync 方法的第一个参数 self 自动绑定到这个 KKNetwork 实例
            # sync 方法的第二个参数 other_network 则是另一个 KKNetwork 实例 KKNetwork(L, N, K, -1)
            # 换句话说，这里同时实例化了两个 KKNetwork 对象，并将它们用于同步操作
            executor.submit(
                KKNetwork(L, N, K, 1).sync,  # 第一个 KKNetwork 对象的 sync 方法
                KKNetwork(L, N, K, -1)  # 作为 sync 方法的参数传递的第二个 KKNetwork 对象
            ) for _ in range(num_runs)
        ]

        return [
            # as_completed 函数会在每个任务完成时生成一个 future 对象
            # future.result() 会阻塞直到任务完成，并返回任务的结果
            future.result() for future in tqdm(as_completed(futures), total=num_runs)
        ]
