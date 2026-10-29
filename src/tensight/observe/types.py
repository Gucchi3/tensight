# @file types.py
# @brief Data containers of the observation (C-02)

from dataclasses import dataclass

import torch

from ..types import QuantValue



@dataclass(frozen=True, eq=False)
class WalkLeaf:
    '''
    #* Hold one tensor that walk_value() found in a nested value.
    #! suffix is what follows the base name of the point, for example "[0][logits]".
    '''
    
    suffix: str
    tensor: torch.Tensor
    quant : QuantValue | None
