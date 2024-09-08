import numpy as np
from .parity_machine import TreeParityMachine as TPM


def sync(tpm1, tpm2, rule='anti_hebbian', state='anti_parallel'):
    """
    Synchronize two TPMs based on the specified rule and state

    Args:
        tpm1 (TreeParityMachine): First TPM instance
        tpm2 (TreeParityMachine): Second TPM instance
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk')
        state (str): Synchronization state ('parallel' or 'anti_parallel')

    Returns:
        int: Number of steps until synchronization
    """
    steps = 0

    while True:
        # Generate random input vector X
        X = np.random.choice([-1, 1], size=(tpm1.K, tpm1.N))

        # Update tau for both TPMs
        tpm1.update_tau(X)
        tpm2.update_tau(X)

        # Check the synchronization state condition
        state_condition = (
            (tpm1.tau * tpm2.tau > 0) if state == 'parallel' 
            else (tpm1.tau * tpm2.tau < 0)
        )
        if state_condition:
            tpm1.update_W(X, rule)
            tpm2.update_W(X, rule)
        
        steps += 1

        if tpm1.is_sync(tpm2, state):  # Check if TPMs are synchronized
            return steps

def sync_with_bit_packages(tpm1, tpm2, B=10, rule='anti_hebbian', state='anti_parallel'):
    """
    Synchronize two TPMs using bit packages based on the specified rule and state

    Args:
        tpm1 (TreeParityMachine): First TPM instance
        tpm2 (TreeParityMachine): Second TPM instance
        B (int): Number of input vectors (bit package size) to process per synchronization step
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk')
        state (str): Synchronization state ('parallel' or 'anti_parallel')

    Returns:
        int: Number of steps until synchronization
    """
    steps = 0

    while True:
        # Generate a batch of B input vectors
        X_batch = np.random.choice([-1, 1], size=(B, tpm1.K, tpm1.N))

        # Lists to store tau and sigma values for both TPMs over the bit package
        tpm1_bit_package = []
        tpm2_bit_package = []
        tpm1_sigma_package = []
        tpm2_sigma_package = []

        # Iterate over the bit package
        for X in X_batch:
            # Update tau and sigma for both TPMs
            tpm1.update_tau(X)
            tpm2.update_tau(X)

            # Store the tau and sigma values for later weight updates
            tpm1_bit_package.append(tpm1.tau)
            tpm2_bit_package.append(tpm2.tau)
            tpm1_sigma_package.append(tpm1.sigma.copy())
            tpm2_sigma_package.append(tpm2.sigma.copy())

        # Now process the bit package and update weights accordingly
        for i in range(B):
            state_condition = (
                (tpm1_bit_package[i] * tpm2_bit_package[i] > 0) if state == 'parallel'
                else (tpm1_bit_package[i] * tpm2_bit_package[i] < 0)
            )
            if state_condition:
                # Update weights using the stored sigma and tau values
                tpm1.update_W(X_batch[i], rule, tpm1_bit_package[i], tpm1_sigma_package[i])
                tpm2.update_W(X_batch[i], rule, tpm2_bit_package[i], tpm2_sigma_package[i])

            steps += 1  # Increment the step count after each input vector is processed

            # Check if the TPMs are synchronized
            if tpm1.is_sync(tpm2, state):
                return steps
