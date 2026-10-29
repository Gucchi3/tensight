# @file sync.py
# @brief Count the GPU-to-CPU syncs
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable

import torch
import warnings

SYNC_MESSAGE = "called a synchronizing CUDA operation"



def sync_count(fn: Callable, *args: object) -> int:
    '''
    #* Run fn(*args) once, and count the GPU-to-CPU syncs that it makes.
    #? Use it to check that your own layer has no syncs. CUDA graphs need zero syncs.
    #! It returns 0 when there is no GPU.
    '''
    
    # There is nothing to count without a GPU
    if not torch.cuda.is_available():
        return 0
    
    # Run fn while PyTorch warns on every sync
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn(*args)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum(1 for warning in caught if SYNC_MESSAGE in str(warning.message))
