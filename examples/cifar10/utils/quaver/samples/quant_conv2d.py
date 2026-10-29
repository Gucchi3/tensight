# @file quant_conv2d.py
# @brief Sample quantized Conv2d
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..formats    import RoundMode
from ..ops        import ActivationKind, accumulate_conv2d
from ..quantizers import Scaler
from ..scales     import QuantParams
from .quant_layer import QuantLayer



class QuantConv2d(QuantLayer):
    '''
    #* Conv2d with quantized weights (one scale for each channel), an INT32 bias, an optional ReLU or ReLU6, and an optional requantization.
    #? forward(x, x_params) returns (output, output params). Without output_format, the output is the INT32 accumulator.
    #! It is not a subclass of nn.Conv2d, because its forward() takes and returns different things. The weight and the bias have the same names.
    '''
    
    output_axis = 1
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, weight_format: str = "int8", output_format: str | None = None, activation: ActivationKind = "none",
                 style: str = "mul16_shift", round_mode: RoundMode | None = None, weight_symmetric: bool = True, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* Make the conv and its quantizers.
        #? activation is "none", "relu", or "relu6". style is how the device requantizes: mul16_shift, shift_only, or exact. output_scaler shares the output scale (a SharedScaler for scale sharing).
        #! weight_symmetric=False or output_symmetric=False gives that quantizer a zero point (INT formats only).
        '''
        
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_format, output_format, activation, style, round_mode, weight_symmetric, output_symmetric, output_scaler)
        self.stride  = stride
        self.padding = padding
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Training and float modes: the float conv on fake-quantized (or float) values.
        '''
        
        return F.conv2d(x, weight, bias, self.stride, self.padding)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* Integer mode: the exact conv accumulator on codes.
        '''
        
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding)
