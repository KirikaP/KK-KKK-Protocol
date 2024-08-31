from kkk_protocol import NeuralNetwork, get_random_inputs, binary_to_hl_outputs
from tqdm import tqdm  # Progress bar library

GENETIC_SYNCHRONISATION_THRESHOLD = 5  # Synchronization threshold
GENETIC_EPOCH_LIMIT = 1200
NUM_RUNS = 1  # Number of runs to calculate average synchronization steps

def run_genetic_attack_kkk_protocol(neuralNetA, neuralNetB, attackNets, inputs, k, n, l, m, sync_threshold, epoch_limit):
    """Run the genetic attack KKK protocol, simulating an attacker's attempt to synchronize with two communication networks."""
    s = 0
    epoch = 0
    
    while s < sync_threshold and epoch < epoch_limit:
        outputA = neuralNetA.get_network_output(inputs)
        outputB = neuralNetB.get_network_output(inputs)

        if outputA == outputB:
            new_attack_nets = []
            hl_outputs = binary_to_hl_outputs(k, outputA)
            for net in attackNets:
                for hl in hl_outputs:
                    cloned_net = NeuralNetwork(k, n, l)
                    cloned_net.weights = [row[:] for row in net.weights]
                    cloned_net.update_weights_given_hl_outputs(inputs, hl)
                    new_attack_nets.append(cloned_net)
            attackNets = new_attack_nets[:m]  # Limit the number of networks

            # Check if any of the attacker's networks have synchronized with A and B
            for net in attackNets:
                if net.get_network_output(inputs) == outputA:
                    s += 1
                    break
            else:
                s = 0  # Reset the counter if no networks are synchronized

        else:
            s = 0  # Reset the counter if A and B outputs differ

        inputs = get_random_inputs(k, n)
        epoch += 1

    return s == sync_threshold, epoch

def main():
    # Set network parameters
    k = 3  # Number of neurons in the hidden layer
    n = 100  # Number of inputs per neuron
    l = 3  # Weight range (from -l to l)
    m = 100  # Maximum number of attack networks

    total_epochs = 0
    successful_runs = 0

    # Run NUM_RUNS times and calculate the average synchronization steps
    for _ in tqdm(range(NUM_RUNS), desc="Running Genetic Attack Simulations"):
        neuralNetA = NeuralNetwork(k, n, l)
        neuralNetB = NeuralNetwork(k, n, l)
        attackNets = [NeuralNetwork(k, n, l)]

        inputs = get_random_inputs(k, n)
        success, epoch = run_genetic_attack_kkk_protocol(neuralNetA, neuralNetB, attackNets, inputs, k, n, l, m, GENETIC_SYNCHRONISATION_THRESHOLD, GENETIC_EPOCH_LIMIT)
        
        if success:
            total_epochs += epoch
            successful_runs += 1

    if successful_runs > 0:
        average_epochs = total_epochs / successful_runs
        print(f"Average synchronization steps across {successful_runs} successful runs: {average_epochs:.2f}")
    else:
        print(f"No successful synchronizations in {NUM_RUNS} runs.")

if __name__ == "__main__":
    main()
