# @file model_outputs.py
# @brief Let torch.export capture a model whose forward() returns codes with their parameters
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import Any

import torch

from ..scales import QuantParams, params_note



class ModelOutputs(nn.Module):
    '''
    #* Run a model, and return its outputs as a tuple of tensors: a tensor stays, and a pair (codes, QuantParams) gives its codes.
    #? model_export() captures this, since torch.export takes only tensors as outputs. The codes of a pair get a note of their parameters (params_note), which the export writes on the output.
    #! forward() of the model may return a tensor, a pair, or a tuple or list of tensors and pairs.
    '''
    
    def __init__(self, model: nn.Module) -> None:
        '''
        #* Keep the model.
        '''
        
        super().__init__()
        self.model = model
        
        return None
    
    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''
        #* Run the model, and keep the tensors and the noted codes of the pairs, in order.
        '''
        
        returned = self.model(*inputs)
        items    = [returned] if isinstance(returned, torch.Tensor) or ModelOutputs.pair(returned) else list(returned)
        return tuple(item if isinstance(item, torch.Tensor) else params_note(*item) for item in items)
    
    @staticmethod
    def pair(item: Any) -> bool:
        '''
        #* Say whether item is a pair of codes and their QuantParams.
        #? forward() uses this to read what the model returns.
        '''
        
        return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], torch.Tensor) and isinstance(item[1], QuantParams)
