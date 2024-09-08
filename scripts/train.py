import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from parity_machine import TreeParityMachine as TPM
from kk_protocol import sync, sync_with_bit_packages
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def tpm_init(L, N, K, zero_replace):
    """
    Create a new instance of TreeParityMachine with the provided parameters

    Args:
        L (int): Weight limit range [-L, L]
        N (int): Number of input bits per hidden unit
        K (int): Number of hidden units
        zero_replace (int): Value to replace 0 in the sigma computation
    
    Returns:
        TreeParityMachine: A new TPM instance.
    """
    return TPM(L, N, K, zero_replace)

def submit_sync_tasks(executor, L, N, K, zero_replace_1, zero_replace_2, rule, state, num_runs, B=None):
    """
    Submit sync tasks to the executor for parallel processing

    Args:
        executor (ProcessPoolExecutor): The executor for parallel processing
        L, N, K (int): Parameters for TPM creation
        zero_replace_1, zero_replace_2 (int): Zero replacement for tpm1 and tpm2
        rule (str): Learning rule for the sync process
        state (str): Synchronization state
        num_runs (int): Number of runs to submit
        B (int or None): Size of the bit package for bit package-based synchronization

    Returns:
        List of Future objects
    """
    futures = []
    for _ in range(num_runs):
        tpm1 = tpm_init(L, N, K, zero_replace_1)
        tpm2 = tpm_init(L, N, K, zero_replace_2)

        # Choose between normal sync or bit package sync based on B
        if B is None:
            future = executor.submit(sync, tpm1, tpm2, rule, state)
        else:
            future = executor.submit(sync_with_bit_packages, tpm1, tpm2, B, rule, state)
        
        futures.append(future)
    return futures

def collect_results(futures, num_runs):
    """
    Collect the results from the completed futures

    Args:
        futures (list of Future): List of submitted futures for processing
        num_runs (int): Total number of runs submitted

    Returns:
        list: A list of sync steps for each run
    """
    results = []
    for future in tqdm(as_completed(futures), total=num_runs):
        result = future.result()
        results.append(result)
    return results

def train_TPMs(
    L, K, N, zero_replace_1, zero_replace_2,
    num_runs=5000, rule='anti_hebbian', state='anti_parallel', max_workers=None, B=None
):
    """
    Train TPMs using multiple parallel runs

    Args:
        L, K, N (int): Parameters for TPM init
        zero_replace_1, zero_replace_2 (int): Zero replacement for tpm1 and tpm2
        num_runs (int): Number of runs to perform
        rule (str): Learning rule to apply
        state (str): Synchronization state to target
        max_workers (int): Max number of workers for parallel processing
        B (int or None): Bit package size. If None, use normal sync logic

    Returns:
        list: A list of steps taken for each synchronization run
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Pass B to control whether to use bit package or regular sync
        futures = submit_sync_tasks(
            executor, L, N, K, zero_replace_1, zero_replace_2, rule, state, num_runs, B
        )
        results = collect_results(futures, num_runs)

    return results
