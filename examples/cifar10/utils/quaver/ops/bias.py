# @file bias.py
# @brief Bias on the scale of the accumulator (INT32, or the wide accumulator of block-scaled tensors)
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import RoundMode, format_quantize, format_to_integer, ste_attach
from ..scales    import QuantParams
from .accumulate import accumulate_format, accumulate_scale



def bias_quantize(bias: torch.Tensor, x_params: QuantParams, w_params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Quantize the bias to the accumulator (INT32, or wide for block-scaled tensors) on its scale. Returns (fake bias, int64 codes).
    #? The training mode adds the fake bias. The integer mode adds the codes to the accumulator.
    #! The bias gets a straight-through gradient, and the scales get none.
    '''
    
    scale         = accumulate_scale(x_params, w_params).detach().double()
    number_format = accumulate_format(x_params, w_params)
    codes         = format_to_integer(format_quantize(bias.detach().double() / scale, number_format, mode), number_format)
    return ste_attach(bias, (codes.double() * scale).float()), codes
