#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <random>
#include <algorithm>

// KKNetwork class simulating a neural network-like structure
class KKNetwork {
public:
    KKNetwork(int L, int N, int K, int zero_replacement)
        : L(L), N(N), K(K), zero_replacement(zero_replacement), Y(K, 0), O(0), gen(std::random_device{}()) {
        
        // Initialize L_list with values from -L to L
        L_list.resize(2 * L + 1);
        std::iota(L_list.begin(), L_list.end(), -L);

        // Initialize random number generator
        std::uniform_int_distribution<> dis(-L, L);

        // Initialize W with random choices from L_list
        W.resize(K, std::vector<int>(N));
        for (auto& row : W) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }

    // Calculate Y based on input vector X
    void get_Y(const std::vector<int>& X) {
        for (int k = 0; k < K; ++k) {
            int sum = std::inner_product(W[k].begin(), W[k].end(), X.begin(), 0);
            Y[k] = sign(sum);
            if (Y[k] == 0) {
                Y[k] = zero_replacement;
            }
        }
    }

    // Calculate the output O
    void get_O() {
        O = std::accumulate(Y.begin(), Y.end(), 1, std::multiplies<int>());
    }

    int get_O_value() const {
        return O;
    }

    // Update the weights based on the input vector X
    void update_weights(const std::vector<int>& X) {
        for (int k = 0; k < K; ++k) {
            if (O * Y[k] > 0) {
                for (int n = 0; n < N; ++n) {
                    W[k][n] -= O * X[n];
                    W[k][n] = std::clamp(W[k][n], -L, L);
                }
            }
        }
    }

    const std::vector<std::vector<int>>& get_weights() const {
        return W;
    }

private:
    int L, N, K, zero_replacement, O;
    std::vector<int> L_list;
    std::vector<std::vector<int>> W;
    std::vector<int> Y;
    std::mt19937 gen;

    int sign(int x) {
        return (x > 0) - (x < 0);
    }
};

// Function to generate a random vector of size N with values either -1 or 1
std::vector<int> generate_random_vector(int N) {
    std::vector<int> X(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, 1);

    for (int& x : X) {
        x = dis(gen) * 2 - 1; // Generates either -1 or 1
    }
    return X;
}

bool weights_are_equal(const std::vector<std::vector<int>>& W1, const std::vector<std::vector<int>>& W2) {
    return W1 == W2;
}

// Perform a single update to synchronize two networks
int single_update(int L, int N, int K) {
    KKNetwork S(L, N, K, 1);
    KKNetwork R(L, N, K, -1);
    int step_count = 0;

    while (true) {
        std::vector<int> X = generate_random_vector(N);

        S.get_Y(X);
        S.get_O();
        R.get_Y(X);
        R.get_O();

        if (S.get_O_value() * R.get_O_value() > 0) {
            S.update_weights(X);
            R.update_weights(X);
        }

        step_count++;

        if (weights_are_equal(S.get_weights(), R.get_weights())) {
            return step_count;
        }
    }
}

// Train the networks and calculate the average synchronization time
std::vector<int> train(int L, int N, int K, int num_runs = 5000) {
    std::vector<int> step_counts;
    step_counts.reserve(num_runs);

    // Create a vector to store futures
    std::vector<std::future<int>> futures;

    // Launch parallel tasks using std::async
    for (int i = 0; i < num_runs; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&, i]() {
            return single_update(L, N, K);
        }));
    }

    // Collect results as they complete
    for (auto& future : futures) {
        step_counts.push_back(future.get());
    }

    return step_counts;
}

int main() {
    int L = 3;
    int K = 3;
    int num_runs = 5000;

    // Define different N values to test
    std::vector<int> N_values = {10, 100, 1000, 10000, 100000, 1000000};

    for (int N : N_values) {
        std::cout << "Running for N = " << N << std::endl;
        std::vector<int> step_counts = train(L, N, K, num_runs);

        // Calculate the average synchronization time (average number of steps)
        double average_steps = std::accumulate(step_counts.begin(), step_counts.end(), 0.0) / step_counts.size();

        // Output the average synchronization time
        std::cout << "N = " << N << ", sync timesteps: " << average_steps << std::endl;
    }

    return 0;
}
