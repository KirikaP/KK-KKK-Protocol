import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM


def sync(tpm1, tpm2, rule='anti_hebbian', state='anti_parallel', return_weights=False):
    """
    Synchronize two Tree Parity Machines (TPMs) using the specified rule and state.

    Args:
        tpm1 (TreeParityMachine): The first TPM.
        tpm2 (TreeParityMachine): The second TPM.
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk').
            Default is 'anti_hebbian'.
        state (str): Synchronization state ('parallel' or 'anti_parallel').
            Default is 'anti_parallel'.
        return_weights (bool): If True, return the weights of tpm1 after synchronization.
            Default is False.

    Returns:
        int: The number of steps until synchronization.
        np.ndarray (optional): The weights of tpm1, if return_weights is True.
    """
    steps = 0
    while True:
        # Generate a random input vector X with values -1 or 1
        X = np.random.choice([-1, 1], size=(tpm1.K, tpm1.N))

        # Update tau values for both TPMs based on the input vector X
        tpm1.update_tau(X)
        tpm2.update_tau(X)

        # Determine if the state condition is met based on the specified state
        state_condition = (
            (tpm1.tau * tpm2.tau > 0) if state == 'parallel' 
            else (tpm1.tau * tpm2.tau < 0)
        )
        if state_condition:
            tpm1.update_W(X, rule)
            tpm2.update_W(X, rule)

        steps += 1

        # Check if the TPMs are synchronized based on the specified state
        if tpm1.is_sync(tpm2, state):
            if return_weights:
                return steps, tpm1.W
            return steps

def sync_with_bit_packages(
    tpm1, tpm2, B=10, rule='anti_hebbian', state='anti_parallel', return_weights=False
):
    """
    Synchronize two Tree Parity Machines (TPMs) using bit packages.

    Args:
        tpm1 (TreeParityMachine): The first TPM.
        tpm2 (TreeParityMachine): The second TPM.
        B (int): Number of input vectors per batch.
            Default is 10.
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk').
            Default is 'anti_hebbian'.
        state (str): Synchronization state ('parallel' or 'anti_parallel').
            Default is 'anti_parallel'.
        return_weights (bool): If True, return the weights of tpm1 after synchronization.
            Default is False.

    Returns:
        int: The number of steps until synchronization.
        np.ndarray (optional): The weights of tpm1, if return_weights is True.
    """
    steps = 0
    while True:
        # Generate a batch of random input vectors X with values -1 or 1
        X_batch = np.random.choice([-1, 1], size=(B, tpm1.K, tpm1.N))

        tpm1_bit_package = []
        tpm2_bit_package = []
        tpm1_sigma_package = []
        tpm2_sigma_package = []

        # For each input vector X, update tau values for tpm1 and tpm2, and save copies of sigma values
        for X in X_batch:
            tpm1.update_tau(X)
            tpm2.update_tau(X)
            tpm1_bit_package.append(tpm1.tau)
            tpm2_bit_package.append(tpm2.tau)
            tpm1_sigma_package.append(tpm1.sigma.copy())
            tpm2_sigma_package.append(tpm2.sigma.copy())

        # For each input vector in the batch, check the state condition and update weights
        for i in range(B):
            state_condition = (
                (tpm1_bit_package[i] * tpm2_bit_package[i] > 0) if state == 'parallel'
                else (tpm1_bit_package[i] * tpm2_bit_package[i] < 0)
            )
            if state_condition:
                tpm1.update_W(X_batch[i], rule, tpm1_bit_package[i], tpm1_sigma_package[i])
                tpm2.update_W(X_batch[i], rule, tpm2_bit_package[i], tpm2_sigma_package[i])
            steps += 1

            # Check if the TPMs are synchronized based on the specified state
            if tpm1.is_sync(tpm2, state):
                if return_weights:
                    return steps, tpm1.W
                return steps
