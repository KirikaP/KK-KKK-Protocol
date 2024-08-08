#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <chrono>

// Simple function to display a progress bar
void display_progress_bar(int current, int total, int bar_width = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

class KKNetwork {
public:
    KKNetwork(int L, int N, int K, int zero_replacement)
        : L(L), N(N), K(K), zero_replacement(zero_replacement), Y(K, 0), O(0) {
        
        // Initialize L_list with values from -L to L
        L_list.resize(2 * L + 1);
        std::iota(L_list.begin(), L_list.end(), -L);

        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-L, L);

        // Initialize W with random choices from L_list
        W.resize(K, std::vector<int>(N));
        for (auto& row : W) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }

    void get_Y(const std::vector<int>& X) {
        for (int k = 0; k < K; ++k) {
            int sum = 0;
            for (int n = 0; n < N; ++n) {
                sum += W[k][n] * X[n];
            }
            Y[k] = sign(sum);
            if (Y[k] == 0) {
                Y[k] = zero_replacement;
            }
        }
    }

    void get_O() {
        O = std::accumulate(Y.begin(), Y.end(), 1, std::multiplies<int>());
    }

    int get_O_value() const {
        return O;
    }

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

    int sign(int x) {
        return (x > 0) - (x < 0);
    }
};

// Function to generate a random vector of size N with values either -1 or 1
std::vector<int> generate_random_vector(int N) {
    std::vector<int> X(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < N; ++i) {
        X[i] = dis(gen) * 2 - 1; // Generates either -1 or 1
    }

    return X;
}

bool weights_are_equal(const std::vector<std::vector<int>>& W1, const std::vector<std::vector<int>>& W2) {
    return W1 == W2;
}

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

std::vector<int> train(int L, int N, int K, int num_runs = 5000) {
    std::vector<int> step_counts;
    step_counts.reserve(num_runs);  // Reserve space to optimize memory allocation

    // Create a vector to store futures
    std::vector<std::future<int>> futures;

    // Mutex for safely updating the progress
    std::mutex progress_mutex;
    int completed = 0;

    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

    // Launch parallel tasks using std::async
    for (int i = 0; i < num_runs; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&, i]() {
            int result = single_update(L, N, K);

            // Safely update progress and display the progress bar
            {
                std::lock_guard<std::mutex> lock(progress_mutex);
                completed++;
                display_progress_bar(completed, num_runs);
            }

            return result;
        }));
    }

    // Collect results as they complete
    for (auto& future : futures) {
        int result = future.get();  // Wait for the result and retrieve it
        step_counts.push_back(result);
    }

    // Ensure the final progress bar shows 100%
    std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> elapsed = end_time - start_time; // Calculate elapsed time

    // Output elapsed time and run speed
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Speed: " << num_runs / elapsed.count() << " runs per second" << std::endl;

    return step_counts;
}

int main() {
    int L = 3;
    int K = 3;
    int num_runs = 5000;  // Number of runs

    // Define different N values to test
    std::vector<int> N_values = {10, 100, 1000, 10000, 100000, 1000000};

    for (int N : N_values) {
        std::cout << "Running for N = " << N << std::endl;
        std::vector<int> step_counts = train(L, N, K, num_runs);

        // Calculate the average synchronization time (average number of steps)
        double average_steps = std::accumulate(step_counts.begin(), step_counts.end(), 0.0) / step_counts.size();

        // Output the average synchronization time
        std::cout << "N = " << N << ", Average synchronization time (in steps): " << average_steps << std::endl;
    }

    return 0;
}