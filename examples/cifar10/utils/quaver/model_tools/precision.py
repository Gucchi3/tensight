# @file precision.py
# @brief Float settings: turn off TF32, and choose the float type of the training-mode math
# @author Kohei Yamaguchi, Claude

import torch

from ..scales import MATH_DTYPES, MathDtype

MATH_TYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}



def precision_disable_tf32() -> None:
    '''
    #* Turn off TF32 in convolutions and matrix products.
    #? cuDNN uses TF32 for float32 convolutions by default, and then the INT32 accumulator cannot be taken back exactly.
    #! This is a setting for the whole process. It cost about 10% of the step time on an RTX 3060.
    '''
    
    # The type hints of torch do not have cudnn.conv, so it is reached with getattr()
    torch.backends.cuda.matmul.fp32_precision            = "ieee"
    getattr(torch.backends.cudnn, "conv").fp32_precision = "ieee"
    
    return None


def precision_set_math(train: MathDtype = "float64", test: MathDtype = "float64") -> None:
    '''
    #* Choose the float type of the training-mode math: train while gradients are on (training steps), and test while they are off (tests in torch.no_grad()).
    #? float32 makes training about 1.6x faster (ResNet-8, with a CUDA graph). A test in float64 still gives the device codes exactly. The integer mode always uses float64.
    #! With float32, the training mode can be one code off the device in rare places. Call it before you capture a CUDA graph (a graph keeps its type).
    '''
    
    # Check the names here at the entry
    for name in (train, test):
        if name not in MATH_TYPES:
            raise KeyError(f"Unknown math type {name!r}. Available: {', '.join(MATH_TYPES)}")
    
    # Set the types for the steps with gradients and for the steps without them
    MATH_DTYPES["train"] = MATH_TYPES[train]
    MATH_DTYPES["test"]  = MATH_TYPES[test]
    
    return None
