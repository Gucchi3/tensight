# @file capture.py
# @brief Whether a CUDA graph is being captured or torch.compile is compiling (ADR-003, ADR-016)

import torch



_cuda: bool | None = None



def capture_active() -> bool:
    '''
    #* Return True while a CUDA graph is being captured or torch.compile is compiling, so that the hooks do nothing (ADR-003, ADR-016).
    #! It does not wait for the GPU. ROCm also uses the torch.cuda functions.
    #! While compiling it reads nothing else: dynamo would make a value that it reads into a guard, and compile again when the value changes.
    '''
    
    # torch.compile is tracing the code
    if torch.compiler.is_compiling():
        return True
    
    # A CUDA graph is being captured
    global _cuda
    if _cuda is None:
        _cuda = torch.cuda.is_available()
    return _cuda and torch.cuda.is_current_stream_capturing()
