from kkk_protocol import NeuralNetwork, get_random_inputs
from tqdm import tqdm  # Progress bar library

EPOCH_LIMIT = 1000000
SYNCHRONISATION_THRESHOLD = 5
NUM_RUNS = 1  # Number of runs to calculate average synchronization steps

def run_geometric_attack_kkk_protocol(neuralNetA, neuralNetB, attackerNet, inputs, k, n, l, sync_threshold, epoch_limit):
    """Run the geometric attack KKK protocol, simulating an attacker's attempt to synchronize with two communication networks."""
    s = 0
    epoch = 0
    sync_epoch_AB = None  # To record the step at which A and B synchronize
    
    while s < sync_threshold and epoch < epoch_limit:
        outputA = neuralNetA.get_network_output(inputs)
        outputB = neuralNetB.get_network_output(inputs)
        outputC = attackerNet.get_network_output(inputs)

        if outputA == outputB == outputC:
            s += 1
            neuralNetA.update_weights(inputs)
            neuralNetB.update_weights(inputs)
            attackerNet.update_weights(inputs)
        elif outputA == outputB:
            s = 0
            if sync_epoch_AB is None:
                sync_epoch_AB = epoch  # Record the step at which A and B synchronize
            min_neuron = attackerNet.get_min_input_sum_neuron(inputs)
            attackerNet.hidden_layer_outputs[min_neuron] *= -1
            attackerNet.update_weights_given_hl_outputs(inputs, attackerNet.hidden_layer_outputs)
            neuralNetA.update_weights(inputs)
            neuralNetB.update_weights(inputs)
        else:
            s = 0
            sync_epoch_AB = None  # Reset the synchronization record if A and B desynchronize

        inputs = get_random_inputs(k, n)
        epoch += 1

    return s == sync_threshold, epoch, sync_epoch_AB

def main():
    # Set network parameters
    k = 3  # Number of neurons in the hidden layer
    n = 32  # Number of inputs per neuron
    l = 4  # Weight range (from -l to l)

    total_epochs = 0
    total_sync_epochs_AB = 0
    successful_runs = 0
    successful_syncs_AB = 0

    # Run NUM_RUNS times and calculate the average synchronization steps
    for _ in tqdm(range(NUM_RUNS), desc="Running Geometric Attack Simulations"):
        neuralNetA = NeuralNetwork(k, n, l)
        neuralNetB = NeuralNetwork(k, n, l)
        neuralNetC = NeuralNetwork(k, n, l)

        inputs = get_random_inputs(k, n)
        success, epoch, sync_epoch_AB = run_geometric_attack_kkk_protocol(neuralNetA, neuralNetB, neuralNetC, inputs, k, n, l, SYNCHRONISATION_THRESHOLD, EPOCH_LIMIT)
        
        if success:
            total_epochs += epoch
            successful_runs += 1
        if sync_epoch_AB is not None:
            total_sync_epochs_AB += sync_epoch_AB
            successful_syncs_AB += 1

    if successful_runs > 0:
        average_epochs = total_epochs / successful_runs
        print(f"Average synchronization steps across {successful_runs} successful runs: {average_epochs:.2f}")
    else:
        print(f"No successful synchronizations in {NUM_RUNS} runs.")

    if successful_syncs_AB > 0:
        average_sync_epochs_AB = total_sync_epochs_AB / successful_syncs_AB
        print(f"Average steps for A and B to synchronize across {successful_syncs_AB} runs: {average_sync_epochs_AB:.2f}")
    else:
        print(f"A and B did not synchronize in any of the {NUM_RUNS} runs.")

if __name__ == "__main__":
    main()
