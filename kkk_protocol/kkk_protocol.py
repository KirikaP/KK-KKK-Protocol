import random
import math

class NeuralNetwork:
    def __init__(self, k, n, l):
        self.k = k  # Number of neurons in the hidden layer
        self.n = n  # Number of inputs per neuron
        self.l = l  # Bound for weight initialization
        self.weights = self._initialize_weights()
        self.hidden_layer_outputs = [0] * k

    def _initialize_weights(self):
        """Initialize weights with random values between -l and l."""
        return [[self._weight_rand() for _ in range(self.n)] for _ in range(self.k)]

    def _weight_rand(self):
        """Generate a random weight between -l and l."""
        return random.randint(-self.l, self.l)

    def update_weights(self, inputs):
        """Update weights using the anti-Hebbian learning rule."""
        hl_outputs = self.get_hidden_layer_outputs(inputs)
        for i in range(self.k):
            for j in range(self.n):
                self.weights[i][j] -= hl_outputs[i] * inputs[i][j]
                self.weights[i][j] = max(min(self.weights[i][j], self.l), -self.l)

    def update_weights_given_hl_outputs(self, inputs, hl_outputs):
        """Update weights with externally supplied hidden layer outputs."""
        for i in range(self.k):
            for j in range(self.n):
                self.weights[i][j] -= hl_outputs[i] * inputs[i][j]
                self.weights[i][j] = max(min(self.weights[i][j], self.l), -self.l)

    def get_hidden_layer_outputs(self, inputs):
        """Compute hidden layer outputs based on the current weights and inputs."""
        for i in range(self.k):
            sum_input = sum(self.weights[i][j] * inputs[i][j] for j in range(self.n))
            self.hidden_layer_outputs[i] = 1 if sum_input > 0 else -1
        return self.hidden_layer_outputs

    def get_network_output(self, inputs):
        """Compute the final output of the network."""
        hl_outputs = self.get_hidden_layer_outputs(inputs)
        return math.prod(hl_outputs)

    def print_weights(self):
        """Print the weights of the network."""
        for i in range(self.k):
            print(', '.join(map(str, self.weights[i])))
        print()

    def get_min_input_sum_neuron(self, inputs):
        """Find the neuron with the minimum input sum."""
        min_sum = float('inf')
        min_neuron = 0
        for i in range(self.k):
            sum_input = sum(self.weights[i][j] * inputs[i][j] for j in range(self.n))
            if abs(sum_input) < min_sum:
                min_sum = abs(sum_input)
                min_neuron = i
        return min_neuron


def get_random_inputs(k, n):
    """Generate random inputs (either -1 or 1) for the network."""
    return [[random.choice([-1, 1]) for _ in range(n)] for _ in range(k)]

def count_zeros(x):
    """Count the number of zero bits in an integer."""
    return bin(x).count('0') - 1

def binary_combinations(k, output):
    """Generate binary combinations that result in the given output."""
    combinations = []
    for i in range(2 ** k):
        if (count_zeros(i) % 2 == 1 and output == -1) or (count_zeros(i) % 2 == 0 and output == 1):
            combinations.append(i)
    return combinations

def binary_to_hl_outputs(k, output):
    """Convert binary combinations to hidden layer outputs."""
    b_combinations = binary_combinations(k, output)
    hl_outputs = []
    for comb in b_combinations:
        hl_outputs.append([1 if (comb >> j) & 1 else -1 for j in range(k)])
    return hl_outputs
