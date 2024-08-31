def get_random_inputs(k, n):
    # 返回形状为 (k, n) 的随机输入数组
    return np.random.choice([-1, 1], (k, n))