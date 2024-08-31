import numpy as np
from tqdm import tqdm  # 如果需要显示进度条

class NeuralNetwork:
    def __init__(self, k, n, l):
        self.k = k
        self.n = n
        self.l = l
        self.weights = np.random.randint(-l, l + 1, (k, n))
        self.hidden_layer_outputs = np.zeros(k, dtype=int)

    def get_hidden_layer_outputs(self, inputs):
        # 计算隐藏层输出
        hl_outputs = np.sign(np.dot(self.weights, inputs.T).sum(axis=1))
        # 将0值替换为-1
        hl_outputs[hl_outputs == 0] = -1
        return hl_outputs

    def get_network_output(self, inputs):
        hl_outputs = self.get_hidden_layer_outputs(inputs)
        return np.prod(hl_outputs)

    def update_weights(self, inputs):
        hl_outputs = self.get_hidden_layer_outputs(inputs)
        for i in range(self.k):
            for j in range(self.n):
                self.weights[i, j] -= hl_outputs[i] * inputs[i, j]
                self.weights[i, j] = np.clip(self.weights[i, j], -self.l, self.l)

def get_random_inputs(k, n):
    return np.random.choice([-1, 1], (k, n))

def run_kkk_protocol(neural_net_a, neural_net_b, k, n, l, epoch_limit):
    s = 0
    epoch = 0
    
    while epoch < epoch_limit:
        inputs = get_random_inputs(k, n)
        output_a = neural_net_a.get_network_output(inputs)
        output_b = neural_net_b.get_network_output(inputs)

        if output_a == output_b:
            s += 1
            neural_net_a.update_weights(inputs)
            neural_net_b.update_weights(inputs)
        else:
            s = 0
        
        epoch += 1

    return epoch  # 返回达到同步所需的轮数

# 示例使用：
k = 3  # 隐藏神经元数量
n = 100  # 每个隐藏神经元的输入数量
l = 3  # 权重范围
epoch_limit = 2000  # 轮次限制
num_experiments = 5000  # 重复实验次数

# 统计同步所需的步数
total_epochs = 0

for _ in tqdm(range(num_experiments), desc="Running experiments"):
    # 创建两个神经网络实例
    neural_net_a = NeuralNetwork(k, n, l)
    neural_net_b = NeuralNetwork(k, n, l)

    # 运行3K协议，记录同步所需的步数
    epochs = run_kkk_protocol(neural_net_a, neural_net_b, k, n, l, epoch_limit)
    total_epochs += epochs

# 计算平均同步所需步数
average_epochs = total_epochs / num_experiments
print(f"平均同步所需步数: {average_epochs:.2f}")
