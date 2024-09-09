import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM
from protocol import sync, sync_with_bit_packages
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def submit_sync_tasks(
    executor, L, N, K, zero_replace_1, zero_replace_2, rule, state, num_runs,
    B=None, return_weights=False
):
    """
    Submit synchronization tasks to be executed in parallel.

    Args:
        executor (ProcessPoolExecutor): Executor for parallel tasks.
        L (int): Weight range limit [-L, L].
        N (int): Number of input bits per hidden unit.
        K (int): Number of hidden units.
        zero_replace_1 (int): Value to replace 0 in the first TPM's sigma calculation.
        zero_replace_2 (int): Value to replace 0 in the second TPM's sigma calculation.
        rule (str): Learning rule ('hebbian', 'anti_hebbian', 'random_walk').
        state (str): Target synchronization state ('parallel' or 'anti_parallel').
        num_runs (int): Number of synchronization runs to execute.
        B (int, optional): Number of input vectors per step (bit-package mode). Defaults to None.
        return_weights (bool, optional): If True, return final weights along with steps. Defaults to False.

    Returns:
        list: A list of futures representing the submitted tasks.
    """
    futures = []
    for _ in range(num_runs):
        tpm1 = TPM(L, N, K, zero_replace_1)
        tpm2 = TPM(L, N, K, zero_replace_2)

        if B is None:
            future = executor.submit(sync, tpm1, tpm2, rule, state, return_weights)
        else:
            future = executor.submit(sync_with_bit_packages, tpm1, tpm2, B, rule, state, return_weights)

        futures.append(future)
    return futures

def collect_results(futures, num_runs, return_weights=False):
    """
    Collect results from completed futures.

    Args:
        futures (list): List of futures to collect results from.
        num_runs (int): Total number of synchronization runs.
        return_weights (bool): If True, collect both steps and final weights.

    Returns:
        list: A list of results, either steps or (steps, weights) depending on return_weights.
    """
    results = []
    for future in tqdm(as_completed(futures), total=num_runs):
        result = future.result()
        results.append(result)
    return results

def train_TPMs(
    L, K, N, zero_replace_1, zero_replace_2,
    num_runs=5000, rule='anti_hebbian', state='anti_parallel',
    max_workers=None, B=None, return_weights=False
):
    """
    Run multiple parallel synchronization tasks.

    Args:
        L (int): Weight range limit [-L, L].
        K (int): Number of hidden units.
        N (int): Number of input bits per hidden unit.
        zero_replace_1 (int): Value to replace 0 in the first TPM's sigma calculation.
        zero_replace_2 (int): Value to replace 0 in the second TPM's sigma calculation.
        num_runs (int): Number of synchronization runs to execute.
        rule (str): Learning rule to apply ('hebbian', 'anti_hebbian', 'random_walk').
        state (str): Target synchronization state ('parallel' or 'anti_parallel').
        max_workers (int, optional): Maximum number of parallel workers. Defaults to None.
        B (int, optional): Number of input vectors per step (bit-package mode). Defaults to None.
        return_weights (bool, optional): If True, return final weights along with steps. Defaults to False.

    Returns:
        list: A list of steps for each synchronization run, or (steps, weights) if return_weights is True.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = submit_sync_tasks(
            executor, L, N, K, zero_replace_1, zero_replace_2, rule, state, num_runs, B, return_weights
        )
        results = collect_results(futures, num_runs, return_weights)

    return results
