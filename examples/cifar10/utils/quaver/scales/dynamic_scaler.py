# @file dynamic_scaler.py
# @brief Scale from the current tensor (dynamic quantization)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat, build_format, device_define
from .compute  import compute_scale
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity, GranularityKind



class DynamicScaler(nn.Module):
    '''
    #* Measure the range of the current tensor every time (dynamic quantization).
    #? Good for weights and for activations with one scale for each token. It keeps no state, so it also measures in inference.
    '''
    
    learnable: bool = False
    
    def __init__(self, percentile: float | None = None, symmetric: bool = True) -> None:
        '''
        #* Keep the settings of the range.
        #? percentile=None uses the maximum.
        '''
        
        super().__init__()
        self.percentile = percentile
        self.symmetric  = symmetric
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the scale and the zero point of x. Both broadcast against x.
        #? The op torch.ops.quaver.dynamic_scale measures them, so an exported model keeps one node with the settings (the device does the same math).
        #! The scale gets no gradient (x is detached), as before.
        '''
        
        return torch.ops.quaver.dynamic_scale(x.detach(), number_format.name, self.percentile, self.symmetric, granularity.kind, granularity.axis, granularity.block_size)


@device_define("dynamic_scale")
def dynamic_scaler_device(x: torch.Tensor, number_format: str, percentile: float | None, symmetric: bool, kind: GranularityKind, axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* The op "dynamic_scale": the scale and the zero point of x from its range now (the maximum, or the percentile), for each group of the granularity.
    #? DynamicScaler.forward() calls it as torch.ops.quaver.dynamic_scale. Both outputs broadcast against x.
    #! The math is float (the range and the division). The device needs the same steps, so this op stays one node in an exported model.
    '''
    
    granularity       = Granularity(kind, axis, block_size)
    low, high         = minmax_measure(x, granularity, percentile, symmetric)
    scale, zero_point = compute_scale(low, high, build_format(number_format), symmetric)
    return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
