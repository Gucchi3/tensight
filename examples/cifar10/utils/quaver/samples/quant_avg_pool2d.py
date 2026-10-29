# @file quant_avg_pool2d.py
# @brief Sample quantized average pooling
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import RoundMode, format_round_mode
from ..ops     import mode_ops
from ..scales  import QuantParams



class QuantAvgPool2d(nn.Module):
    '''
    #* Average pooling on integer codes: add, divide by the count, and round.
    #? forward(x, x_params) returns (output, x_params). The scale does not change.
    '''
    
    round_mode: RoundMode | None
    
    def __init__(self, kernel_size: int, stride: int | None = None, round_mode: RoundMode | None = None) -> None:
        '''
        #* Keep the settings of the pooling.
        #? round_mode=None uses the usual rounding of the input format (half_away for INT). FP inputs always round half to even.
        '''
        
        super().__init__()
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.round_mode   = round_mode
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Pool x in the current mode (fake values, integer codes, or float values).
        '''
        
        ops = mode_ops(self.integer_mode, self.float_mode)
        return ops.pool_avg2d(x, x_params, self.kernel_size, self.stride, format_round_mode(x_params.number_format, self.round_mode)), x_params
