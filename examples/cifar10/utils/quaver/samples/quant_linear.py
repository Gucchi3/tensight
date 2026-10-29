# @file quant_linear.py
# @brief Sample quantized Linear
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..formats    import RoundMode
from ..ops        import ActivationKind, accumulate_linear
from ..quantizers import Scaler
from ..scales     import QuantParams
from .quant_layer import QuantLayer



class QuantLinear(QuantLayer):
    '''
    #* Linear with quantized weights (one scale for each channel), an INT32 bias, an optional ReLU or ReLU6, and an optional requantization.
    #? forward(x, x_params) returns (output, output params). Without output_format, the output is the INT32 accumulator.
    #! It is not a subclass of nn.Linear, because its forward() takes and returns different things. The weight and the bias have the same names.
    '''
    
    output_axis = -1
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_format: str = "int8", output_format: str | None = None, activation: ActivationKind = "none", style: str = "mul16_shift", round_mode: RoundMode | None = None,
                 weight_symmetric: bool = True, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* Make the linear layer and its quantizers.
        #? activation is "none", "relu", or "relu6". style is how the device requantizes: mul16_shift, shift_only, or exact. output_scaler shares the output scale (a SharedScaler for scale sharing).
        #! weight_symmetric=False or output_symmetric=False gives that quantizer a zero point (INT formats only).
        '''
        
        super().__init__(nn.Linear(in_features, out_features, bias=bias), weight_format, output_format, activation, style, round_mode, weight_symmetric, output_symmetric, output_scaler)
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Training and float modes: the float linear op on fake-quantized (or float) values.
        '''
        
        return F.linear(x, weight, bias)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* Integer mode: the exact linear accumulator on codes.
        '''
        
        return accumulate_linear(q_x, x_params, q_w, w_params)
