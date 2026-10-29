# @file quant_block_linear.py
# @brief Sample block-scaled Linear (NVFP4 and the like), with integer math only
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..ops              import ActivationKind, accumulate_linear
from ..quantizers       import Scaler
from ..scales           import QuantParams
from .quant_block_layer import QuantBlockLayer



class QuantBlockLinear(QuantBlockLayer):
    '''
    #* Linear with block-scaled weights (a scale for each block of 16 input features), optional block-scaled inputs, an exact wide accumulator, and the output in the format of the next op.
    #? fp4 weights with scale_format "ue4m3" are NVFP4. With INT8 inputs (input_format=None) it is W4A8. forward(x, x_params) returns (output, output params).
    #! It is not a subclass of nn.Linear, because its forward() takes and returns different things. The weight and the bias have the same names.
    '''
    
    output_axis = -1
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_format: str = "fp4", scale_format: str = "ue4m3", input_format: str | None = None, output_format: str = "int8", output_block: bool = False,
                 activation: ActivationKind = "none", style: str = "mul16_shift", block_size: int = 16, output_scaler: Scaler | None = None) -> None:
        '''
        #* Make the linear layer and its quantizers.
        #? input_format=None takes the input as it is (block-scaled, or INT8). output_block=True gives a block-scaled output in output_format, for the next block-scaled layer.
        #? output_scaler shares the scale of the INT output (output_block=False), like the residual stream of a block (scale sharing).
        #! in_features must be a multiple of block_size, and so must out_features with output_block=True.
        '''
        
        super().__init__(nn.Linear(in_features, out_features, bias=bias), weight_format, scale_format, input_format, output_format, output_block, activation, style, block_size, output_scaler)
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Training and float modes: the float linear op on fake-quantized (or float) values.
        '''
        
        return F.linear(x, weight, bias)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* The exact linear accumulator on codes (each code times the code of its block scale).
        '''
        
        return accumulate_linear(q_x, x_params, q_w, w_params)
